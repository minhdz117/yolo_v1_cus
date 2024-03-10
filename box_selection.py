import torch

def non_max_supression(boxes, threshold, p_threshold):
    """
    boxes = [class, probability, box ... ]
    """
    most_prob_boxes = sorted(
        [b for b in boxes if b[1] > p_threshold], key=lambda x: x[1], reverse=True
    )
    cleand_boxes = []
    while most_prob_boxes:
        current_box = most_prob_boxes.pop(0)

        most_prob_boxes = [
            box
            for box in most_prob_boxes
            if box[0] != current_box[0]
            or IoU(torch.tensor(box[2:]), torch.tensor(current_box[2:])) < threshold
        ]

        cleand_boxes.append(current_box)

    return cleand_boxes
