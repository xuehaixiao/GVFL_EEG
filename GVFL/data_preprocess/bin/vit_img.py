import scipy.io as sio
import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel
from transformers import AutoImageProcessor,AutoModelForImageClassification
import random
random.seed(12)
device = "cuda" if torch.cuda.is_available() else "cpu"


# 加载预训练的ViT-L模型和特征提取器
local_model_path ='/media/ubuntu/EEDE7473DE743645/EEG40000/CVPR2021-02785//models--google--vit-large-patch16-224-in21k/snapshots/6074eaf2211423e928c93b93ef773d5da618aa7e'
image_processor = ViTImageProcessor.from_pretrained(local_model_path)
model = ViTModel.from_pretrained(local_model_path)
# image_processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-224")
# model = AutoModelForImageClassification.from_pretrained("google/vit-large-patch16-224")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
#Load model directly

model = model.to(device)
model.eval()
# 定义图像预处理步骤
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


classes = {
           "n02106662": 0,
           "n02124075": 1,
           "n02281787": 2,
           "n02389026": 3,
           "n02492035": 4,
           "n02504458": 5,
           "n02510455": 6,
           "n02607072": 7,
           "n02690373": 8,
           "n02906734": 9,
           "n02951358": 10,
           "n02992529": 11,
           "n03063599": 12,
           "n03100240": 13,
           "n03180011": 14,
           "n03272010": 15,
           "n03272562": 16,
           "n03297495": 17,
           "n03376595": 18,
           "n03445777": 19,
           "n03452741": 20,
           "n03584829": 21,
           "n03590841": 22,
           "n03709823": 23,
           "n03773504": 24,
           "n03775071": 25,
           "n03792782": 26,
           "n03792972": 27,
           "n03877472": 28,
           "n03888257": 29,
           "n03982430": 30,
           "n04044716": 31,
           "n04069434": 32,
           "n04086273": 33,
           "n04120489": 34,
           "n04555897": 35,
           "n07753592": 36,
           "n07873807": 37,
           "n11939491": 38,
           "n13054560": 39}

iv = "image"
data_path = "/media/ubuntu/EEDE7473DE743645/EEG40000/CVPR2021-02785/00_unpreprocess_0.2_7/"
img_path = "/media/ubuntu/EEDE7473DE743645/EEG40000/CVPR2021-02785/stimuli/"
eegnames = "/home/ubuntu/桌面/xhx/Data/eeg_vit_0.2_4.csv"
imgnames = "/home/ubuntu/桌面/xhx/Data/img_vit_0.2_4.csv"

EEGCVPR = []
ImgCVPR = []

fileList = os.listdir(data_path)
length = len(fileList)
count = 0

print("数据处理中")
for f in tqdm(fileList, desc="Processing files", total=len(fileList)):
    if iv == "image":
        c = f.split("_")[0]
        if f.find('_')==-1:
            continue
        name = f.split(".")[0]
        tmpdata = sio.loadmat(data_path + "/" + f)
        tmpdata = tmpdata["eeg"]
        tmpdata = torch.from_numpy(tmpdata).type(torch.FloatTensor)

        tmplabel = classes[name.split("_")[0]]

        # 获取图像特征
        tmpimgpath = os.path.join(img_path, name + '.JPEG')
        tmpimg = Image.open(tmpimgpath).convert('RGB')
        tmpimg = preprocess().unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model(**tmpimg).pooler_output#.cpu()

        tmpIMG = {"image": image_features, "label": tmplabel}
        ImgCVPR.append(tmpIMG)

        tmpEEG = {"eeg": tmpdata, "label": tmplabel}
        EEGCVPR.append(tmpEEG)

    count += 1

torch.save(EEGCVPR, eegnames)
torch.save(ImgCVPR, imgnames)