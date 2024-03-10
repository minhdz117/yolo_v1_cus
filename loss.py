import torch.nn as nn
import torch
from metrics import IoU

class Yolo_loss(nn.Module):
    def __init__(self, c=6):
        super().__init__()
        self.c = c
        self.mse = nn.MSELoss(reduction="sum")
        self.l_coord = 5
        self.l_noobj = 0.5

    def forward(self, predictions, labels):
        predictions = predictions.reshape(-1, 7, 7, 5 * 2 + self.c )

        obj_exist = labels[..., 0].unsqueeze(3)

        # Box predicion loss
        true_bb = labels[..., 2:6]

        bb1 = predictions[..., 1:5]
        bb2 = predictions[..., 6:10]

        IoU_for_bb1 = IoU(bb1, true_bb)
        IoU_for_bb2 = IoU(bb2, true_bb)

        IoUs = torch.cat([IoU_for_bb1.unsqueeze(0), IoU_for_bb2.unsqueeze(0)], dim=0)
        best_bb, best_bb_ind = torch.max(IoUs, dim=0)

        box_pred = obj_exist * ((1 - best_bb_ind) * bb1 + best_bb_ind * bb2)

        box_center_loss = self.mse(box_pred[..., 0:2], obj_exist * labels[..., 1:3])

        bb_shape_pred = torch.sign(box_pred[..., 2:4]) * torch.sqrt(
            torch.abs(box_pred[..., 2:4] + 1e-5)
        )
        box_shape_loss = self.mse(
            bb_shape_pred, obj_exist * torch.sqrt(labels[..., 3:5])
        )

        # object loss
        obj_pred = obj_exist * (
            (1 - best_bb_ind) * predictions[..., 0:1]
            + best_bb_ind * predictions[..., 5:6]
        )
        obj_true = obj_exist * labels[..., 0:1]

        obj_loss = self.mse(obj_pred, obj_true)

        # no object loss
        noobj_pred_bb1 = (1 - obj_exist) * predictions[..., 0:1]
        noobj_pred_bb2 = (1 - obj_exist) * predictions[..., 5:6]
        noobj_true = (1 - obj_exist) * labels[..., 0:1]

        noobj_loss_bb1 = self.mse(noobj_pred_bb1, noobj_true)
        noobj_loss_bb2 = self.mse(noobj_pred_bb2, noobj_true)

        noobj_loss = noobj_loss_bb1 + noobj_loss_bb2

        # class loss
        pred_classes = predictions[..., 10:17]
        true_classes = labels[..., 5:12]

        class_loss = self.mse(obj_exist * pred_classes, obj_exist * true_classes)

        # total loss
        loss = (
            self.l_coord * box_center_loss
            + self.l_coord * box_shape_loss
            + obj_loss
            + self.l_noobj * noobj_loss
            + class_loss
        )

        return loss
