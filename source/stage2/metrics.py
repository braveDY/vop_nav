import torch


def interval_terms(prediction, target):
    low, high = prediction.unbind(-1)
    truth_low, truth_high = target.unbind(-1)
    intersection = (torch.minimum(high, truth_high) - torch.maximum(low, truth_low)).clamp_min(0)
    predicted_length = (high - low).clamp_min(0)
    true_length = (truth_high - truth_low).clamp_min(0)
    union = predicted_length + true_length - intersection
    # Two empty sets match. Empty versus nonempty has zero overlap.
    iou = torch.where(union <= 1e-8, torch.ones_like(union), intersection / union.clamp_min(1e-8))
    return iou, intersection, predicted_length, true_length


def vop_loss(prediction, target, alpha=.2):
    iou = interval_terms(prediction, target)[0]
    mse = (prediction - target).square().mean()
    iou_loss = (1 - iou).mean()
    return alpha * iou_loss + (1 - alpha) * mse, {'iou_loss': iou_loss, 'mse': mse}


@torch.no_grad()
def batch_metrics(prediction, target, alpha=.2):
    iou, intersection, predicted, truth = interval_terms(prediction, target)
    loss, terms = vop_loss(prediction, target, alpha)
    return dict(loss=float(loss), iou=float(iou.mean()), mse=float(terms['mse']),
                boundary_mae_mps=float((prediction - target).abs().mean()),
                unsafe_length_mps=float((predicted - intersection).mean()),
                missed_safe_length_mps=float((truth - intersection).mean()))
