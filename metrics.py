import torch

def IoU(box_pred, box_true):
    """
    Itersection over Union
    box = [x, y, width, height]
    where x, y - middle point coordinates

    """
    pred_widths, pred_heights = box_pred[..., 2:3], box_pred[..., 3:4]
    true_widths, true_heights = box_true[..., 2:3], box_true[..., 3:4]

    box_pred_x1, box_pred_y1 = (
        box_pred[..., 0:1] - pred_widths / 2,
        box_pred[..., 1:2] - pred_heights / 2,
    )
    box_true_x1, box_true_y1 = (
        box_true[..., 0:1] - true_widths / 2,
        box_true[..., 1:2] - true_heights / 2,
    )

    box_pred_x2, box_pred_y2 = (box_pred_x1 + pred_widths, box_pred_y1 + pred_heights)
    box_true_x2, box_true_y2 = (box_true_x1 + true_widths, box_true_y1 + true_heights)

    intersection_x1 = torch.max(box_pred_x1, box_true_x1)
    intersection_y1 = torch.max(box_pred_y1, box_true_y1)
    intersection_x2 = torch.min(box_pred_x2, box_true_x2)
    intersection_y2 = torch.min(box_pred_y2, box_true_y2)

    intersection = (intersection_x2 - intersection_x1).clamp(0) * (
        intersection_y2 - intersection_y1).clamp(0)


    box_pred_area = (box_pred_x2 - box_pred_x1) * (box_pred_y2 - box_pred_y1)
    box_true_area = (box_true_x2 - box_true_x1) * (box_true_y2 - box_true_y1)

    union = box_pred_area + box_true_area - intersection

    return intersection / (union + 1e-5)
