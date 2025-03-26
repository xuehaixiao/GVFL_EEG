import random
import torch
from torch.utils.data import DataLoader, random_split, Subset
from GVFL.gvfl_classification.LoaderCon import CusEEGDataset,get_img_features
data='/home/ubuntu/桌面/xhx/Data/eeg_vit.csv'
image='/home/ubuntu/桌面/xhx/Data/img_vit.csv'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def meanclass():
    dataset = CusEEGDataset('all', data, image, device)

    labels = [dataset.eeg[i]["label"] for i in range(len(dataset))]  # 假设dataset[i]返回(data, label)
    indices = list(range(len(dataset)))

    num_classes = 40  # 假设有10个类别
    samples_per_class = 1000

    # 根据标签分组
    class_indices = {i: [] for i in range(num_classes)}
    for idx, label in zip(indices, labels):
        if label < num_classes and len(class_indices[label]) < samples_per_class:
            class_indices[label].append(idx)

    # 检查是否每个类别都有1000个样本
    assert all(len(indices) == samples_per_class for indices in class_indices.values()), "Not all classes have 1000 samples"

    # 分配比例
    train_ratio = 0.6
    val_ratio = 0.3
    test_ratio = 0.1

    # 计算每个类别中每个子集应有的样本数
    train_samples_per_class = int(train_ratio * samples_per_class)
    val_samples_per_class = int(val_ratio * samples_per_class)
    test_samples_per_class = samples_per_class - train_samples_per_class - val_samples_per_class

    # 为每个类别创建子集
    train_indices, val_indices, test_indices = [], [], []
    for class_label, class_idxs in class_indices.items():
        random.shuffle(class_idxs)  # 打乱该类别的索引
        train_class_idxs = class_idxs[:train_samples_per_class]
        val_class_idxs = class_idxs[train_samples_per_class:train_samples_per_class + val_samples_per_class]
        test_class_idxs = class_idxs[train_samples_per_class + val_samples_per_class:]
        train_indices.extend(train_class_idxs)
        val_indices.extend(val_class_idxs)
        test_indices.extend(test_class_idxs)

    # 创建子集数据集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, val_loader, test_loader