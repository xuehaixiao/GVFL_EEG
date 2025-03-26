import scipy.io as sio
import argparse
import torch
import os
import random
import pickle as pkl

random.seed(12)

parser = argparse.ArgumentParser(description = "pkl path")
parser.add_argument("-iv",
                    "--iv",
                    help = "image/video",
                    type = str,
                    required = True)
parser.add_argument("-s",
                    "--size",
                    help = "small/big",
                    type = str,
                    required = True)
parser.add_argument("-p",
                    "--dataset-path",
                    help = "dataset-path",
                    type = str,
                    required = True)
parser.add_argument("-l",
                    "--label-path",
                    help = "label-path",
                    type = str,
                    required = False)
parser.add_argument("-n",
                    "--name",
                    help = "name",
                    type = str,
                    required = True)
parser.add_argument("-stimuli",
                    "--stimuli",
                    help = "stimuli",
                    type = str,
                    required = True)
parser.add_argument("-f",
                    "--fold",
                    help = "number of folds",
                    type = int,
                    required = True)
# I set f 5 for image, and -f 4 for video

classes = {"n02106662": 0,
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

args = parser.parse_args()

data_path = args.dataset_path
label_path = args.label_path
DataCVPR = [];
LabelCVPR = [];

if args.iv=="image":
    f = open(args.stimuli, "r")
    stimuli = [line.split(".")[0] for line in f.readlines()]
    f.close()
if args.iv=="video":
    f = open(args.stimuli, "r")
    stimuli = [line.split("\n")[0] for line in f.readlines()]
    f.close()

fileList = os.listdir(data_path)
if args.size=="small":
    label = pkl.load(open(label_path,"r"))
count = 0

Class = {}
for f in fileList:
    if args.iv=="image":
        c = f.split("_")[0]
    if args.iv=="video":
        c = f.split("-")[0]
    name = f.split(".")[0]
    tmpdata = sio.loadmat(data_path+"/"+f)
    tmpdata = tmpdata["eeg"]
    tmpdata = torch.from_numpy(tmpdata).type(torch.FloatTensor)
    if args.size=="small":
        tmplabel = int(label[name])
    if args.size=="big":
        tmplabel = classes[name.split("_")[0]]
    LabelCVPR.append(tmplabel)
    tmpDic = {"eeg": tmpdata, "label": tmplabel, "image": stimuli.index(name)}
    DataCVPR.append(tmpDic)
    if c not in Class:
        Class[c] = [count]
    else:
        Class[c].append(count)
    count += 1

mydict = {"dataset": DataCVPR, "labels": LabelCVPR, "images": stimuli}
dataset_name = args.name
torch.save(mydict, dataset_name)

split = []

length = len(fileList)
fold_num = args.fold

if args.iv=="image":
    number_of_classes = 40

if args.iv=="video":
    number_of_classes = 12

for k, v in Class.items():
    sample = random.sample(range(len(v)), len(v))
    Class[k] = [v[i] for i in sample]

samples_per_class = length/number_of_classes
samples_per_class_per_fold = samples_per_class/fold_num
half_samples_per_class_per_fold = samples_per_class_per_fold/2

for i in range(fold_num):
    spliti = {"train": [],"val": [],"test": []}
    sample = [k for k in range(samples_per_class)]
    subsample1 =  [k for k
                   in range(i*samples_per_class_per_fold,
                            i*samples_per_class_per_fold+
                            half_samples_per_class_per_fold)]
    subsample2 =  [k for k
                   in range(i*samples_per_class_per_fold+
                            half_samples_per_class_per_fold,
                            (i+1)*samples_per_class_per_fold)]
    for k, v in Class.items():
        for j in sample:
            if j in subsample1:
                spliti["val"].append(v[j])
            elif j in subsample2:
                spliti["test"].append(v[j])
            else:
                spliti["train"].append(v[j])
    split.append(spliti)

mydict = {"splits": split}
dataset_name = args.name.split(".")[0]+"_split.pth"
torch.save(mydict, dataset_name)
