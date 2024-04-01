"""
Generate statistics for mesh reconstruction results.
Copied from https://github.com/ErlerPhilipp/points2surf/blob/master/source/base/evaluation.py.
"""

import os
import subprocess
import multiprocessing

import numpy as np
import hydra
from omegaconf import DictConfig


def make_dir_for_file(file):
    """
    Make dir for file.
    """
    file_dir = os.path.dirname(file)
    if file_dir != '':
        if not os.path.exists(file_dir):
            try:
                os.makedirs(os.path.dirname(file))
            except OSError as exc:  # Guard against race condition
                raise


def mp_worker(call):
    """
    Small function that starts a new thread with a system call. Used for thread pooling.
    """
    call = call.split(' ')
    verbose = call[-1] == '--verbose'
    if verbose:
        call = call[:-1]
        subprocess.run(call)
    else:
        # subprocess.run(call, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # suppress outputs
        subprocess.run(call, stdout=subprocess.DEVNULL)


def start_process_pool(worker_function, parameters, num_processes, timeout=None):

    if len(parameters) > 0:
        if num_processes <= 1:
            print('Running loop for {} with {} calls on {} workers'.format(
                str(worker_function), len(parameters), num_processes))
            results = []
            for c in parameters:
                results.append(worker_function(*c))
            return results
        print('Running loop for {} with {} calls on {} subprocess workers'.format(
            str(worker_function), len(parameters), num_processes))
        with multiprocessing.Pool(processes=num_processes, maxtasksperchild=1) as pool:
            results = pool.starmap(worker_function, parameters)
            return results
    else:
        return None


def _chamfer_distance_single_file(file_in, file_ref, samples_per_model, num_processes=1):
    # http://graphics.stanford.edu/courses/cs468-17-spring/LectureSlides/L14%20-%203d%20deep%20learning%20on%20point%20cloud%20representation%20(analysis).pdf

    import trimesh
    import trimesh.sample
    import sys
    import scipy.spatial as spatial

    def sample_mesh(mesh_file, num_samples):
        try:
            mesh = trimesh.load(mesh_file)
        except:
            return np.zeros((0, 3))
        samples, face_indices = trimesh.sample.sample_surface_even(mesh, num_samples)
        return samples

    try:
        new_mesh_samples = sample_mesh(file_in, samples_per_model)
        ref_mesh_samples = sample_mesh(file_ref, samples_per_model)
    except AttributeError:
        # unable to sample
        return file_in, file_ref, -1.0

    if new_mesh_samples.shape[0] == 0 or ref_mesh_samples.shape[0] == 0:
        return file_in, file_ref, -1.0

    leaf_size = 100
    sys.setrecursionlimit(int(max(1000, round(new_mesh_samples.shape[0] / leaf_size))))
    kdtree_new_mesh_samples = spatial.cKDTree(new_mesh_samples, leaf_size)
    kdtree_ref_mesh_samples = spatial.cKDTree(ref_mesh_samples, leaf_size)

    ref_new_dist, corr_new_ids = kdtree_new_mesh_samples.query(ref_mesh_samples, 1, workers=num_processes)
    new_ref_dist, corr_ref_ids = kdtree_ref_mesh_samples.query(new_mesh_samples, 1, workers=num_processes)

    ref_new_dist_sum = np.sum(ref_new_dist)
    new_ref_dist_sum = np.sum(new_ref_dist)
    chamfer_dist = ref_new_dist_sum + new_ref_dist_sum

    return file_in, file_ref, chamfer_dist


def _hausdorff_distance_directed_single_file(file_in, file_ref, samples_per_model):
    import scipy.spatial as spatial
    import trimesh
    import trimesh.sample

    def sample_mesh(mesh_file, num_samples):
        try:
            mesh = trimesh.load(mesh_file)
        except:
            return np.zeros((0, 3))
        samples, face_indices = trimesh.sample.sample_surface_even(mesh, num_samples)
        return samples

    try:
        new_mesh_samples = sample_mesh(file_in, samples_per_model)
        ref_mesh_samples = sample_mesh(file_ref, samples_per_model)
    except AttributeError:
        # unable to sample
        return file_in, file_ref, -1.0

    if new_mesh_samples.shape[0] == 0 or ref_mesh_samples.shape[0] == 0:
        return file_in, file_ref, -1.0

    dist, _, _ = spatial.distance.directed_hausdorff(new_mesh_samples, ref_mesh_samples)
    return file_in, file_ref, dist


def _hausdorff_distance_single_file(file_in, file_ref, samples_per_model):
    import scipy.spatial as spatial
    import trimesh
    import trimesh.sample

    def sample_mesh(mesh_file, num_samples):
        try:
            mesh = trimesh.load(mesh_file)
        except:
            return np.zeros((0, 3))
        samples, face_indices = trimesh.sample.sample_surface_even(mesh, num_samples)
        return samples

    try:
        new_mesh_samples = sample_mesh(file_in, samples_per_model)
        ref_mesh_samples = sample_mesh(file_ref, samples_per_model)
    except AttributeError:
        # unable to sample
        return file_in, file_ref, -1.0, -1.0, -1.0

    if new_mesh_samples.shape[0] == 0 or ref_mesh_samples.shape[0] == 0:
        return file_in, file_ref, -1.0, -1.0, -1.0

    dist_new_ref, _, _ = spatial.distance.directed_hausdorff(new_mesh_samples, ref_mesh_samples)
    dist_ref_new, _, _ = spatial.distance.directed_hausdorff(ref_mesh_samples, new_mesh_samples)
    dist = max(dist_new_ref, dist_ref_new)
    return file_in, file_ref, dist_new_ref, dist_ref_new, dist


def _scale_single_file(file_ref):
    import trimesh
    if not file_ref.endswith('.obj'):
        file_ref = file_ref + '.obj'
    mesh = trimesh.load(file_ref)
    extents = mesh.extents
    scale = extents.max()
    return scale


def mesh_comparison(new_meshes_dir_abs, ref_meshes_dir_abs,
                    num_processes, report_name, samples_per_model=10000, dataset_file_abs=None):
    if not os.path.isdir(new_meshes_dir_abs):
        print('Warning: dir to check doesn\'t exist'.format(new_meshes_dir_abs))
        return

    new_mesh_files = [f for f in os.listdir(new_meshes_dir_abs)
                      if os.path.isfile(os.path.join(new_meshes_dir_abs, f))]
    ref_mesh_files = [f for f in os.listdir(ref_meshes_dir_abs)
                      if os.path.isfile(os.path.join(ref_meshes_dir_abs, f))]

    if dataset_file_abs is None:
        mesh_files_to_compare_set = set(ref_mesh_files)  # set for efficient search
    else:
        if not os.path.isfile(dataset_file_abs):
            raise ValueError('File does not exist: {}'.format(dataset_file_abs))
        with open(dataset_file_abs) as f:
            mesh_files_to_compare_set = f.readlines()
            mesh_files_to_compare_set = [f.replace('\n', '') + '.ply' for f in mesh_files_to_compare_set]
            mesh_files_to_compare_set = [f.split('.')[0] for f in mesh_files_to_compare_set]
            mesh_files_to_compare_set = set(mesh_files_to_compare_set)

    # # skip if everything is unchanged
    # new_mesh_files_abs = [os.path.join(new_meshes_dir_abs, f) for f in new_mesh_files]
    # ref_mesh_files_abs = [os.path.join(ref_meshes_dir_abs, f) for f in ref_mesh_files]
    # if not utils_files.call_necessary(new_mesh_files_abs + ref_mesh_files_abs, report_name):
    #     return

    def ref_mesh_for_new_mesh(new_mesh_file: str, all_ref_meshes: list) -> list:
        stem_new_mesh_file = new_mesh_file.split('.')[0]
        ref_files = list(set([f for f in all_ref_meshes if f.split('.')[0] == stem_new_mesh_file]))
        return ref_files

    call_params = []
    for fi, new_mesh_file in enumerate(new_mesh_files):
        if new_mesh_file.split('.')[0] in mesh_files_to_compare_set:
            new_mesh_file_abs = os.path.join(new_meshes_dir_abs, new_mesh_file)
            ref_mesh_files_matching = ref_mesh_for_new_mesh(new_mesh_file, ref_mesh_files)
            if len(ref_mesh_files_matching) > 0:
                ref_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_mesh_files_matching[0])
                call_params.append((new_mesh_file_abs, ref_mesh_file_abs, samples_per_model))
    if len(call_params) == 0:
        raise ValueError('Results are empty!')
    results_hausdorff = start_process_pool(_hausdorff_distance_single_file, call_params, num_processes)
    results = [(r[0], r[1], str(r[2]), str(r[3]), str(r[4])) for r in results_hausdorff]

    call_params = []
    for fi, new_mesh_file in enumerate(new_mesh_files):
        if new_mesh_file.split('.')[0] in mesh_files_to_compare_set:
            new_mesh_file_abs = os.path.join(new_meshes_dir_abs, new_mesh_file)
            ref_mesh_files_matching = ref_mesh_for_new_mesh(new_mesh_file, ref_mesh_files)
            if len(ref_mesh_files_matching) > 0:
                ref_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_mesh_files_matching[0])
                call_params.append((new_mesh_file_abs, ref_mesh_file_abs, samples_per_model))
    results_chamfer = start_process_pool(_chamfer_distance_single_file, call_params, num_processes)
    results = [r + (str(results_chamfer[ri][2]),) for ri, r in enumerate(results)]

    # no reference but reconstruction
    for fi, new_mesh_file in enumerate(new_mesh_files):
        if new_mesh_file.split('.')[0] not in mesh_files_to_compare_set:
            if dataset_file_abs is None:
                new_mesh_file_abs = os.path.join(new_meshes_dir_abs, new_mesh_file)
                ref_mesh_files_matching = ref_mesh_for_new_mesh(new_mesh_file, ref_mesh_files)
                if len(ref_mesh_files_matching) > 0:
                    reference_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_mesh_files_matching[0])
                    results.append((new_mesh_file_abs, reference_mesh_file_abs, str(-2), str(-2), str(-2), str(-2)))
        else:
            mesh_files_to_compare_set.remove(new_mesh_file.split('.')[0])

    # no reconstruction but reference
    for ref_without_new_mesh in mesh_files_to_compare_set:
        new_mesh_file_abs = os.path.join(new_meshes_dir_abs, ref_without_new_mesh)
        reference_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_without_new_mesh)
        results.append((new_mesh_file_abs, reference_mesh_file_abs, str(-1), str(-1), str(-1), str(-1)))

    # append scale to each row
    call_params = []
    for fi, row in enumerate(results):
        ref_file_abs = row[1]
        call_params.append([ref_file_abs])
    results_scale = start_process_pool(_scale_single_file, call_params, num_processes)
    results = [r + (str(results_scale[ri]),) for ri, r in enumerate(results)]

    # sort by file name
    results = sorted(results, key=lambda x: x[0])

    make_dir_for_file(report_name)
    csv_lines = ['in mesh,ref mesh,Hausdorff dist new-ref,Hausdorff dist ref-new,Hausdorff dist,'
                 'Chamfer dist(-1: no input; -2: no reference),Scale']
    csv_lines += [','.join(item) for item in results]
    # csv_lines += ['=AVERAGE(E2:E41)']
    csv_lines_str = '\n'.join(csv_lines)
    with open(report_name, "w") as text_file:
        text_file.write(csv_lines_str)


@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def generate_stats(cfg: DictConfig):
    """
    Evaluate Hausdorff distance between reconstructed and GT models.

    Parameters
    ----------
    cfg: DictConfig
        Hydra configuration
    """

    csv_file = os.path.join(cfg.csv_path)
    mesh_comparison(
        new_meshes_dir_abs=cfg.remap_dir,
        ref_meshes_dir_abs=cfg.reference_dir,
        num_processes=cfg.num_workers,
        report_name=csv_file,
        samples_per_model=cfg.evaluate.num_samples,
        dataset_file_abs=os.path.join(cfg.data_dir, 'raw/testset.txt'))


if __name__ == '__main__':
    generate_stats()
