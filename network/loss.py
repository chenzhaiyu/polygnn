"""
Loss functions.
"""

import torch
import torch.nn.functional as F


def bce_loss(inputs, targets, weight=(1., 1.)):
    """
    Binary cross entropy loss for PolyGNN prediction.
    """
    pred = inputs.argmax(dim=1)
    total = len(pred)
    correct = torch.sum((pred.clone().detach() == targets).long())
    accuracy = correct / total
    loss = F.nll_loss(inputs, targets, weight=torch.tensor(weight, device=inputs.device))
    ratio = torch.sum(pred) / total
    return loss, accuracy, ratio, total, correct


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Source from https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensors loss, accuracy and ratio.
    """
    ce_loss = F.nll_loss(inputs, targets, reduction="none")
    p_t = torch.exp(-ce_loss)
    p_t = torch.clamp(p_t, max=1.0)  # clip to avoid NaN
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )

    pred = inputs.argmax(dim=1)
    total = len(pred)
    correct = torch.sum((pred.clone().detach() == targets).long())
    accuracy = correct / total
    ratio = torch.sum(pred) / total
    return loss, accuracy, ratio, total, correct
