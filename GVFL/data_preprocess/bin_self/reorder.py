import torch

def reorder_data_by_label(eegpath, imgpath, device):
    # 加载 EEG 和图像数据
    eeg = torch.load(eegpath, map_location=device)
    img = torch.load(imgpath, map_location=device)

    # 获取 EEG 和图像数据中的标签（假设“label”字段包含类别信息）
    eeg_data = [eeg[i]["eeg"] for i in range(len(eeg))]
    eeg_labels = [eeg[i]["label"] for i in range(len(eeg))]
    img_data = [img[i]["img"] for i in range(len(img))]
    img_labels = [img[i]["label"] for i in range(len(img))]

    # 确保 EEG 和图像数据的标签一一对应
    assert eeg_labels == img_labels, "EEG and image labels don't match!"

    # 创建一个字典，按标签将数据分组
    data_dict = {i: {'eeg': [], 'img': []} for i in range(40)}  # 假设标签是0到39

    for i, label in enumerate(eeg_labels):
        data_dict[label]['eeg'].append(eeg_data[i])
        data_dict[label]['img'].append(img_data[i])

    # 按标签从0到39排序并组合数据
    sorted_eeg_data = []
    sorted_img_data = []
    for label in range(40):
        sorted_eeg_data.extend(data_dict[label]['eeg'])
        sorted_img_data.extend(data_dict[label]['img'])

    # 生成新的 EEG 和图像数据
    new_eeg = [{'eeg': eeg, 'label': label} for label, eeg in zip(range(40), sorted_eeg_data)]
    new_img = [{'img': img, 'label': label} for label, img in zip(range(40), sorted_img_data)]

    # 返回排序后的 EEG 和图像数据
    return new_eeg, new_img

# 使用示例
eegpath = '/home/ubuntu/桌面/xhx/Data/eeg_vit_reorder.csv'
imgpath = '/home/ubuntu/桌面/xhx/Data/img_vit_reorder.csv'
device = 'cuda'  # 或者 'cpu'
new_eeg, new_img = reorder_data_by_label(eegpath, imgpath, device)

# 可以将 `new_eeg` 和 `new_img` 保存回文件
torch.save(new_eeg, eegpath)
torch.save(new_img, imgpath)
