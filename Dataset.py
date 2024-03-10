import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from torch.utils.data import Dataset
import torch
import cv2


class Pascal_Images(Dataset):
    def __init__(self, data):

        super().__init__()
        self.num_classes = 6
        self.data = data
        self.enc = OneHotEncoder(handle_unknown="ignore")
        self.enc.fit(np.array([i for i in range(3)]).reshape(-1, 1))
        self.label_encoder = LabelEncoder()

    def __len__(self):
        return len(self.data)  # <---

    def __getitem__(self, index):
        # Grid 7 x 7
        # image size 448 x 448

        image_file = self.data.iloc[index].iloc[0]
        image_path = "Data/image/" + image_file

        data_file = self.data.iloc[index].iloc[1]
        data_path = "Data/label/" + data_file

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (448, 448))
        image = np.transpose(image, (2, 0, 1))

        image_data = pd.read_csv(data_path, header=None, sep=" ")
        # image_data = [class, x, y, bb_x, bb_y]
        cols = image_data.columns
        # scale center coordinates
        # image_data[cols[2]] *= 448
        # image_data[cols[3]] *= 448
        image_data[cols[4]] /= 448
        image_data[cols[5]] /= 448

        # x coordinates relative to the cell
        image_data["rel_x"] = image_data[cols[2]] / 64 % 1
        # y coordinates relative to the cell
        image_data["rel_y"] = image_data[cols[3]] / 64 % 1
        # which cell contains object center (x axis)
        image_data["cell_x"] = image_data[cols[2]] // 64
        # which cell contains object center (y axis)
        image_data["cell_y"] = image_data[cols[3]] // 64
        # p_c if there is object in the cell
        image_data["p_c"] = 1
        image_data[cols[0]] = self.label_encoder.fit_transform(image_data[cols[0]])
        image_data[cols[1]] = self.label_encoder.fit_transform(image_data[cols[1]])

        classes0 = image_data[cols[0]].values.reshape(-1, 1)
        classes1 = image_data[cols[1]].values.reshape(-1, 1)

        ohe = self.enc.transform(classes0).toarray()
        ohe_df = pd.DataFrame(ohe, columns=["class_" + str(i) for i in range(3)])
        image_data = pd.concat((image_data, ohe_df), axis=1)
        ohe = self.enc.transform(classes1).toarray()
        ohe_df = pd.DataFrame(ohe, columns=["class_" + str(i+3) for i in range(3)])
        image_data = pd.concat((image_data, ohe_df), axis=1)
        # columns to extract for the labels:
        extract_cols = ["p_c", "rel_x", "rel_y", cols[4], cols[5]]
        extract_cols.extend(image_data.columns[-self.num_classes:])
        data_array = image_data[extract_cols].to_numpy()

        # indexes for responsible cells
        idx_x = image_data[["cell_x"]].astype(int).to_numpy().flatten()
        idx_y = image_data[["cell_y"]].astype(int).to_numpy().flatten()

        # final label
        # 5 - predictions for each grid cell (p_c,
        # rel_x, rel_y, width, height)
        # 1 - bounding boxes for each cell
        # 6 - classes
        labels = np.zeros((7, 7, 5 * 1 + self.num_classes))
        for i, (_x, _y) in enumerate(zip(idx_x, idx_y)):
            labels[_x, _y] = data_array[i]

        return torch.tensor(image).float(), labels
