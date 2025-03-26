import torch
import copy

import pandas as pd
import numpy as np

import torch
from sklearn.model_selection import train_test_split


def meanclass(eegpath, imgpath, device):
    # 加载 EEG 和图像数据
    eeg = torch.load(eegpath, map_location=device)
    img = torch.load(imgpath, map_location=device)

    # 获取 EEG 和图像数据中的标签（假设“label”字段包含类别信息）
    eeg_data = [eeg[i]["eeg"] for i in range(len(eeg))]
    eeg_labels = [eeg[i]["label"] for i in range(len(eeg))]
    img_data = [img[i]["img"] for i in range(len(img))]
    img_labels = [img[i]["label"] for i in range(len(img))]

    # 确保标签一致
    assert eeg_labels == img_labels, "EEG and image labels don't match!"

    # 将数据按类别划分
    unique_labels = list(set(eeg_labels))  # 获取所有唯一的类别
    train_data, val_data, test_data = [], [], []

    # 对每个类别分别处理
    for label in unique_labels:
        # 获取该类别的所有样本索引
        eeg_idx = [i for i, lbl in enumerate(eeg_labels) if lbl == label]

        # 通过索引获取对应的 EEG 和图像数据
        eeg_class_data = [eeg_data[i] for i in eeg_idx]
        img_class_data = [img_data[i] for i in eeg_idx]

        # 按照 6:3:1 的比例划分数据集
        eeg_train, eeg_temp, img_train, img_temp = train_test_split(
            eeg_class_data, img_class_data, test_size=0.4, random_state=42, stratify=[eeg_labels[i] for i in eeg_idx]
        )
        eeg_val, eeg_test, img_val, img_test = train_test_split(
            eeg_temp, img_temp, test_size=0.25, random_state=42, stratify=[eeg_labels[i] for i in eeg_idx]
        )

        # 将该类别的划分结果加入到总的数据集中
        train_data.extend(list(zip(eeg_train, img_train)))
        val_data.extend(list(zip(eeg_val, img_val)))
        test_data.extend(list(zip(eeg_test, img_test)))

    return train_data, val_data, test_data


class CusEEGDataset:

    def __init__(self, topo, eegpath, imgpath, device):
        self.topo = "all" if topo is None else topo
        self.electrode_regions = {

            'frontal': ['C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17',
                        'C18','C19', 'C20', 'C21', 'C25', 'C26', 'C27', 'C28', 'C29','C30', 'C31', 'C32'],
            'central': ['A1', 'B1', 'B20', 'B21', 'B22', 'B23', 'B24', 'B29', 'B30', 'B31', 'B32','C1', 'C2', 'C3',
                        'C11','C22','C23','C24'],
            'parietal': ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A18', 'A19', 'A31', 'A32', 'B2', 'B3', 'B4',
                         'B10', 'B11', 'B12', 'B13', 'B16', 'B17', 'B18'],
            'temporal': ['B15', 'B14', 'B25', 'B26', 'B27', 'B28'],
            'occipital': ['A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17','A21', 'A22', 'A23',
                          'A24','A25', 'A26', 'A27', 'A28', 'A29', 'A30','B5', 'B6', 'B7', 'B8', 'B9','A20']
        }
        data1020 = pd.read_excel(
            '/media/ubuntu/EEDE7473DE743645/xhx/EEG40000/sensor_dataframe.xlsx',
            index_col=0)
        channels1020 = np.array(data1020.index)[:96]
        self.ch_names = channels1020.tolist()#[::-1]
        self.eeg = torch.load(eegpath, map_location=device)
        self.img = torch.load(imgpath, map_location=device)
        self.size = len(self.eeg)
        self.electrodes = self.electrode_regions.get(self.topo.lower(), [])  # Ensure topo is in lowercase to match keys
        self.indices = [self.ch_names.index(elec) for elec in self.electrodes if elec in self.ch_names]

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        # Process EEG
        labeldata = self.eeg[i]["label"]
        # label = self.data[i]["label"]
        if "image" in self.img[i]:
            imgdata = self.img[i]["image"]
        elif "img" in self.img[i]:
            imgdata = self.img[i]["img"]
        #imgdata = self.img.get("image") or self.img.get("img")
        img_label = self.img[i]["label"]
        if self.topo == "all":
            eegdata = self.eeg[i]["eeg"][:512,]
            #eegdata = torch.rand(eegdata.size())
        else:
            eegdata = self.eeg[i]["eeg"]
            eegdata=eegdata[:, self.indices]
        # im_label = self.img.iloc[i]["label"]
        return eegdata, labeldata, imgdata


class CusEEG_randomDataset:
    def __init__(self,eeg, img, randomize_labels):

        self.size = len(eeg)
        self.eeg = eeg
        self.img = img
        # 处理标签（是否随机化）
        self.randomize_labels = randomize_labels
        self.labels = [self.eeg[i]["label"] for i in range(self.size)]
        if self.randomize_labels:
            self.labels = np.random.permutation(self.labels)  # 随机打乱标签

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        # EEG数据处理
        labeldata = self.labels[i]  # 使用随机或真实标签
        if "image" in self.img[i]:
            imgdata = self.img[i]["image"]
        elif "img" in self.img[i]:
            imgdata = self.img[i]["img"]
        img_label = self.img[i]["label"]
        eegdata = self.eeg[i]["eeg"][:, ]

        return eegdata, labeldata, imgdata


def  get_img_features(imgpath):
    im_loaded = torch.load(imgpath, map_location='cpu')
    category_features = {i: [] for i in range(40)}

    for idx,data in enumerate(im_loaded):
        label = data["label"]
        # if "image" in data:
        images = data["image"]
        # elif "img" in data:
        #images = data["img"]
        category_features[label].append(images)
    del im_loaded
    sorted_img_features = []
    for i in range(40):
        if category_features[i]:  # 确保该类别有特征
            center = np.mean(np.array(category_features[i]), axis=0)
            sorted_img_features.append(center)
    sorted_img_features = np.array(sorted_img_features)
    sorted_saw_label = np.array(range(40))
    return torch.tensor(sorted_img_features).squeeze().float(), torch.tensor(sorted_saw_label)

