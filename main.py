import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataset import Pascal_Images
from model import Yolo_v1
from loss import Yolo_loss
from metrics import IoU
def get_pic(label):
    centr_x_rel = label[..., 1]
    centr_y_rel = label[..., 2]
    width_rel = label[..., 3]
    heigth_rel = label[..., 4]
    cell_x = np.argmax(centr_x_rel) // 7
    cell_y = np.argmax(centr_y_rel) % 7
    center_x_bb = centr_x_rel[cell_x, cell_y]
    center_y_bb = centr_y_rel[cell_x, cell_y]
    width = width_rel[cell_x, cell_y] * 448
    heigth = heigth_rel[cell_x, cell_y] * 448

    center_x = 64 * (cell_x + center_x_bb)
    center_y = 64 * (cell_y + center_y_bb)
    x_min, x_max = center_x - width / 2, center_x + width / 2
    y_min, y_max = center_y - heigth / 2, center_y + heigth / 2

    return (
        center_x,
        center_y,
        [x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min],
    )


def get_bb_center_cells(label):
    object_cells = []
    # print("=", label.shape[0], label.shape[1], "=")
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            # print(i, j, label[i][j])
            if any(label[i][j] != 0):
                object_cells.append((i, j))
    return object_cells


def get_boxes(labels, object_cells):
    bboxes = []
    for _cell in object_cells:
        i, j = _cell
        centr_x_rel = labels[i][j][1]
        centr_y_rel = labels[i][j][2]
        width_rel = labels[i][j][3]
        heigth_rel = labels[i][j][4]

        center_x = 64 * (i + centr_x_rel)
        center_y = 64 * (j + centr_y_rel)

        width = width_rel * 448
        heigth = heigth_rel * 448

        x_min, x_max = center_x - width / 2, center_x + width / 2
        y_min, y_max = center_y - heigth / 2, center_y + heigth / 2
        bboxes.append(
            [
                center_x,
                center_y,
                [x_min, x_max, x_max, x_min, x_min],
                [y_min, y_min, y_max, y_max, y_min],
                width,
                heigth,
            ]
        )
    return bboxes


def get_boxes_wh(labels, object_cells):
    bboxes = []
    for _cell in object_cells:
        i, j = _cell
        centr_x_rel = labels[i][j][1]
        centr_y_rel = labels[i][j][2]
        width_rel = labels[i][j][3]
        heigth_rel = labels[i][j][4]

        center_x = 64 * (i + centr_x_rel)
        center_y = 64 * (j + centr_y_rel)

        width = width_rel * 448
        heigth = heigth_rel * 448

        bboxes.append([labels[i][j][0], center_x, center_y, width, heigth])
    return bboxes


def get_xy(width, height):
    x_min, x_max = center_x - width / 2, center_x + width / 2
    y_min, y_max = center_y - height / 2, center_y + height / 2
    return [x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min]

# add mean avg precision
def train_epoch(model, optimizer, criterion):
    loss_log = []
    for x_batch, y_batch in train_loader:  # <-
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        pred = model(x_batch)

        loss = criterion(pred, y_batch)

        loss_log.append(loss.item())

        loss.backward()

        optimizer.step()

    return loss_log


@torch.no_grad()
def test_epoch(model, criterion):
    loss_log = []
    for x_batch, y_batch in test_loader:  # <-
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        pred = model(x_batch)

        loss = criterion(pred, y_batch)

        loss_log.append(loss.item())

    return loss_log


def train(model, optimizer, criterion, epochs):
    for epoch in range(epochs):
        train_loss = train_epoch(model, optimizer, criterion)
        # test_loss = test_epoch(model, criterion)

        print(
            "epoch: ",
            epoch,
            " | ",
            "train loss: ",
            np.mean(train_loss),
            " | ",
        )
        # "test loss: ", np.mean(test_loss))


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Data directory
    # There is only image and labels file names.
    # Actual files are in folder '/content'.
    train_path = "Data/content/train.csv"
    test_path = "Data/content/val.csv"

    # train/test dataframe
    file_trian = pd.read_csv(train_path)
    file_test = pd.read_csv(test_path)

    # Dataset and DataLoader
    train_dataset = Pascal_Images(file_trian)
    train_loader = DataLoader(
        train_dataset, pin_memory=True, batch_size=8, shuffle=True, drop_last=True
    )

    test_dataset = Pascal_Images(file_test)
    test_loader = DataLoader(
        test_dataset, pin_memory=True, batch_size=8, shuffle=True, drop_last=True
    )

    # Model and parameteres
    model = Yolo_v1()
    model.to(device)
    WEIGHT_DECAY = 0
    optimizer = optim.Adam(
        params=model.parameters(), weight_decay=WEIGHT_DECAY, lr=2e-5
    )  # <-
    criterion = Yolo_loss()
    # for overfitting

    epochs = 400

    train(model, optimizer, criterion, epochs)
