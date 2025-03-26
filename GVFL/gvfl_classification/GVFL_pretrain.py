import os
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
import matplotlib.pyplot as plt
# from t_sne import draw_tsne,draw_tsne_eeg
import torch.nn as nn
from LoaderCon import CusEEGDataset,get_img_features
import random
import torch
from torch.utils.data import DataLoader,random_split
import csv
import itertools
import numpy as np
from EEGNet import EEGNet
from ATMS_50 import ATMS_50
from encoder import Model,Proj_img
from NICE import GA_NICE,SA_NICE,Deepnet_NICE,Shallownet_NICE
from EEGChannelnet import EEGChannelnet
#from s4_addencoder.encoder_s4de import Model_s4
from classify_head import ClassifierHead
from marix_draw import  ConfusionMatrix

def train_model(eeg_model,dataloader, optimizer, device,all_imgclass):
    eeg_model.train()

    total_loss = 0
    total = 0

    features_list = []  # List to store features
    labels_list = []

    for batch_idx, (eeg_data, labels, img_features) in enumerate(dataloader):

        eeg_data = eeg_data.to(device)
        # eeg_data = torch.rand(eeg_data.size())  # .to(device)
        # mean = eeg_data.mean()
        # std = eeg_data.std()
        # eeg_data = ((eeg_data - mean) / std).to(device)
        img_features = img_features.float().squeeze().to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        eeg_features = eeg_model(eeg_data)

        # print("eeg_features", torch.std(eeg_features))
        features_list.append(eeg_features.float())
        labels_list.append(labels)
        logit_scale = eeg_model.logit_scale
        
        loss1 = eeg_model.loss_func(eeg_features.float(), img_features, logit_scale)
        loss = loss1
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # logits = logit_scale * eeg_features @ text_features_all.T # (n_batch, n_cs)
        # logits_img = logit_scale * eeg_features @ all_imgclass.T
        # logits_single = logits_img
        # predicted = torch.argmax(logits_single, dim=1) # (n_batch, ) \in {0, 1, ..., n_cls-1}

        batch_size = labels.shape[0]
        total += batch_size
    average_loss = total_loss / (batch_idx+1)
    return average_loss, torch.cat(features_list, dim=0),torch.cat(labels_list, dim=0)

def evaluate_model(eeg_model, dataloader, device):

    eeg_model.eval()

    features_list = []
    labels_list = []
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (eeg_data, labels, img_features) in enumerate(dataloader):

            eeg_data = eeg_data.to(device)
            #    # .to(device)
            # mean = eeg_data.mean()
            # std = eeg_data.std()
            # eeg_data = ((eeg_data - mean) / std).to(device)
            labels = labels.to(device)
            img_features = img_features.squeeze().float().to(device)#.view(16,768).

            eeg_features = eeg_model(eeg_data)

            features_list.append(eeg_features.float())
            labels_list.append(labels)
            logit_scale = eeg_model.logit_scale 
            # print(eeg_features.type, text_features.type, img_features.type)
            loss1 = eeg_model.loss_func(eeg_features.float(), img_features, logit_scale)
            loss = loss1
            
            total_loss += loss.item()

    average_loss = total_loss / (batch_idx+1)
    return average_loss, torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)


def test_model(eeg_model, dataloader, device, all_imgclass, best_model_weights, test):
    if test:
        eeg_model.load_state_dict(best_model_weights)
    eeg_model.eval()
    features_list = []
    labels_list = []
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    all_imgclass = all_imgclass.to(device)
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            labels = labels.to(device)

            eeg_features = eeg_model(eeg_data)

            features_list.append(eeg_features.float())
            labels_list.append(labels)
            logit_scale = eeg_model.logit_scale
            logits_img = logit_scale * eeg_features @ all_imgclass.T
            logits_single = logits_img
            predicted_probs, predicted = torch.topk(logits_single, 3, dim=1)  # 获取前3个预测及其概率  
            batch_size = predicted.shape[0]
            total += batch_size

            # 计算Top-1准确率  
            correct_top1 += (predicted[:, 0] == labels).sum().item()

            # 计算Top-3准确率  
            for i in range(batch_size):
                if labels[i] in predicted[i]:
                    correct_top3 += 1

        test_top1 = correct_top1 / total
        test_top3 = correct_top3 / total  # 计算Top-3准确率  

    return test_top1, test_top3, torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)



def main_train_loop(topo,current_time, eeg_encoder, train_dataloader, val_dataloader, optimizer, device,
             config,epochs,all_imgclass,all_imglabel):

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    best_val_loss = 100

    best_model_weights = eeg_encoder.state_dict()
    best_epoch_info = {}
    results = []  # List to store results for each epoch
    
    for epoch in range(epochs):
        
        train_loss, _, _ = train_model(eeg_encoder,train_dataloader, optimizer, device,all_imgclass)
        train_losses.append(train_loss)
        val_loss, _, _ = evaluate_model(eeg_encoder, val_dataloader, device)
        val_losses.append(val_loss)

        # Append results for this epoch
        epoch_results = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss
        }

        results.append(epoch_results)

        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            best_model_weights = eeg_encoder.state_dict()

            print('best_epoch:', best_epoch)


            
        best_epoch_info = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        }


        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f},  Val Loss: {val_loss:.4f}")
  
    # TEST
    test_top1, test_top3,eegfeature, eeglabel = test_model(eeg_encoder, val_dataloader, device, all_imgclass, best_model_weights=None, test=False)
    print(f"Test TOP1: {test_top1:.4f},Test TOP3: {test_top3:.4f}")
    # save best model
    modeldir = f"/media/ubuntu/EEDE7473DE743645/xhx/GVFL/gvfl_classification/models/contrast/{config['encoder_type']}/retrieval"
    os.makedirs(modeldir, exist_ok=True)
    file_path = f"{modeldir}/tm-{config['time']}_topo-{config['topo']}_hz-{config['frequency']}_{current_time}.pth"

    os.environ['MODEL_PATH'] = file_path
    torch.save(best_model_weights, file_path)
    print(f"model saved in {file_path}!")
    
    # torch.save(model.state_dict(), '{train_pos_img_text}.pth')
    import matplotlib
    matplotlib.use('Agg')
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 15))

    
    axs[0].plot(train_losses, label='Train Loss')
    axs[0].plot(val_losses, label='Val  Loss')
    axs[0].legend()
    axs[0].set_title("Loss Curve")



    
    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                f"Val Loss: {best_epoch_info['val_loss']:.4f}\n")

    axs[1].axis('off')
    axs[1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[1].transAxes)

    plt.tight_layout()

    plotdir = f"./figure/{config['encoder_type']}/pretrain"
    os.makedirs(plotdir, exist_ok=True)
    plotpath = f"{plotdir}/tm-{config['time']}_topo-{config['topo']}_hz-{config['frequency']}_{current_time}.png"
    plt.savefig(plotpath)
    #draw_tsne_eeg(train_eegfeature, train_eeglabel,state='train')
    #draw_tsne_eeg(eegfeature_4, eeglabel_4,state='test')
    return results,file_path

import datetime
import json
from thop import profile
#from GVFL.data_preprocess.bin_self.meanclass import meanclass
def main():
    seed_n = np.random.randint(2023)
    print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)



    config_file_path = "/media/ubuntu/EEDE7473DE743645/xhx/GVFL/gvfl_classification/config.json"

    with open(config_file_path, 'r') as f:
        config = json.load(f)
    config["seed"] = seed_n

    # 将更新后的配置写回文件
    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # lr=0.0003
    epochs =1
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    # Re-initialize the models for each subject
    eeg_model = globals()[config['encoder_type']](config['strategy'],num_channels=config['channel'][config['topo']])

    if config['encoder_type']=='EEGChannelnet':
        eeg_model.cuda()
    else:
        eeg_model.to(device)
    optimizer = torch.optim.AdamW(itertools.chain(eeg_model.parameters()), lr=3e-4)
    print(f'Processing number of parameters:', sum([p.numel() for p in itertools.chain(eeg_model.parameters())]))
    # input = torch.randn(64,512,96).to(device)
    # flops,params = profile(eeg_model, inputs=(input,), verbose=False)
    # print("FLOPs = "+str(flops/1000**3)+'G')
    # print("Params = "+str(params/1000**2)+'M')

    all_imgclass, all_imglabel = get_img_features(config['image_path'])

    dataset = CusEEGDataset(config['topo'], config['data_path'], config['image_path'], device)
    # 获取数据集的长度
    dataset_size = len(dataset)
    train_size = int(0.8*dataset_size)
    val_size = int(0.2* dataset_size)
    #test_size = int(0.1 * dataset_size)
    #train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size,val_size, test_size])

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    #test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    results,file_path = main_train_loop(config['topo'], current_time, eeg_model, train_loader, val_loader,  optimizer, device,
                                config,epochs,all_imgclass,all_imglabel)

    config["pretrain_model"] = file_path
    config["seed"] = seed_n

    # 将更新后的配置写回文件
    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Save results to a CSV file
    results_dir = f"./outputs/{config['encoder_type']}/retrieval"
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/tm-{config['time']}_topo-{config['topo']}_hz-{config['frequency']}_{current_time}.csv"

    with open(results_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'Results saved to {results_file}')
    # // "data_path": "/home/ubuntu/\u684c\u9762/xhx/Data/eeg_vit_0.2_4.csv",
    # // "image_path": "/home/ubuntu/\u684c\u9762/xhx/Data/img_vit_0.2_4.csv",
            
if __name__ == '__main__':
    main()