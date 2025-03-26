import os
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
import matplotlib.pyplot as plt
# from t_sne import draw_tsne,draw_tsne_eeg
import torch.nn as nn
from LoaderCon import CusEEGDataset,get_img_features
import random
from torch.utils.data import DataLoader,random_split
import csv
import itertools
import numpy as np
import torch
from GVFL.models.EEGNet import EEGNet
from GVFL.models.ATMS_50 import ATMS_50
from GVFL.models.encoder import Model,Proj_img
from GVFL.models.NICE import GA_NICE,SA_NICE
from GVFL.models.EEGChannelnet import EEGChannelnet


def train_model(eeg_model, img_model, dataloader, optimizer, scheduler, device, all_imgclass):
    eeg_model.train()
    img_model.train()

    total_loss = 0
    correct_retrieval = 0
    correct_classification = 0
    total = 0
    alpha=0.99
    features_list = []  # List to store features
    labels_list = []
    criteon = nn.CrossEntropyLoss()
    all_imgclass=all_imgclass.to(device)
    for batch_idx, (eeg_data, labels, img_features) in enumerate(dataloader):

        eeg_data = eeg_data.to(device)
        img_features = img_features.float().squeeze().to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        eeg_features, eeg_class = eeg_model(eeg_data)
        img_features = img_model(img_features).float()
        
        logit_scale = eeg_model.logit_scale
        loss1 = eeg_model.loss_func(eeg_features, img_features, logit_scale)
        loss2 = criteon(eeg_class,labels)
        loss = loss1+loss2
        loss.backward()
        optimizer.step()
        
        features_list.append(eeg_features)
        labels_list.append(labels)

        total_loss += loss.item()
        batch_size = labels.shape[0]
        total += batch_size

            #准确个数
        _, pred = eeg_class.data.max(1)
        correct_classification += pred.eq(labels.data).sum().item()
        
    average_loss = total_loss / (batch_idx+1)

    correct_classification = correct_classification / total
    return average_loss,correct_classification , torch.cat(features_list, dim=0),torch.cat(labels_list, dim=0)

def evaluate_model(eeg_model, img_model, dataloader, device):
    eeg_model.eval()
    img_model.eval()
    features_list = []
    labels_list = []
    total_loss = 0
    correct_classification = 0
    total = 0

    criteon = nn.CrossEntropyLoss()

    with (torch.no_grad()):
        for batch_idx, (eeg_data, labels, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            labels = labels.to(device)
            img_features = img_features.squeeze().float().to(device)#.view(16,768).
            
            eeg_features,eeg_class = eeg_model(eeg_data)
            img_features = img_model(img_features).float()
            features_list.append(eeg_features)
            labels_list.append(labels)
            
            logit_scale = eeg_model.logit_scale 
            loss1 = eeg_model.loss_func(eeg_features, img_features, logit_scale)
            loss2 = criteon(eeg_class, labels)
            loss = loss1+loss2
            total_loss += loss.item()

            batch_size = labels.shape[0]
            total += batch_size
            #准确个数
            _, pred = eeg_class.data.max(1)
            correct_classification += pred.eq(labels.data).sum().item()
        
    average_loss = total_loss / (batch_idx+1)
    correct_classification = correct_classification / total
    return average_loss, correct_classification , torch.cat(features_list, dim=0),torch.cat(labels_list, dim=0)
def load_pretrained_eeg_model(path,strategy,model,channel):

    pretrained_weights = torch.load(path)

    eeg_model = model(strategy,num_channels=channel)
    eeg_model.load_state_dict(pretrained_weights)
    return eeg_model
def main_train_loop(topo, current_time, eeg_model,train_dataloader, val_dataloader,test_dataloader, optimizer, scheduler, device, all_imgclass,all_label,config):

    train_losses, train_classification_accuracies = [], []
    val_losses, val_classification_accuracies = [], []

    best_accuracy = 0.0
    best_model_weights = None
    best_epoch_info = {}
    results = []  # List to store results for each epoch
    
    for epoch in range(config['epochs']):
        #train
        train_loss,  train_classification_accuracy,  train_eegfeature, train_eeglabel = train_model(
            eeg_model, train_dataloader, optimizer,scheduler, device,all_imgclass)
        train_losses.append(train_loss)
        train_classification_accuracies.append(train_classification_accuracy)
        #val
        val_loss, val_classification_accuracy, eegfeature, eeglabel = evaluate_model(
            eeg_model, val_dataloader, device, all_imgclass, all_label)
        val_losses.append(val_loss)
        val_classification_accuracies.append(val_classification_accuracy)

        
        # Append results for this epoch
        epoch_results = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_classification_accuracy": train_classification_accuracy,
        "val_loss": val_loss,
        "val_classification_accuracy": val_classification_accuracy
        }

        results.append(epoch_results)
        #save best model weights
        if val_classification_accuracy > best_accuracy:
            best_accuracy = val_classification_accuracy
            best_model_weights = eeg_model.state_dict().copy()
            
            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_classification_accuracy": train_classification_accuracy,
                "val_loss": val_loss,
                "val_classification_accuracy": val_classification_accuracy
            }

        print(f"Epoch {epoch + 1}/{config['epochs']} - Train Loss: {train_loss:.4f}, Train_classification_Accuracy: {train_classification_accuracy:.4f}")
        print(f"Epoch {epoch + 1}/{config['epochs']} - Val Loss: {val_loss:.4f},Val_classification_Accuracy: {val_classification_accuracy:.4f}")
    #test
    _, test_classification_accuracy, eegfeature, eeglabel = evaluate_model(eeg_model, test_dataloader, device, all_imgclass, all_label)
    print(f"Test_classification_Accuracy: {test_classification_accuracy:.4f}")

    # model.load_state_dict(best_model_weights)
    os.makedirs(f"./models/contrast/{config['encoder_type']}/R+C", exist_ok=True)
    file_path = f"./models/contrast/{config['encoder_type']}/R+C/{current_time}.pth"
    torch.save(best_model_weights, file_path)
    print(f"model saved in {file_path}!")
    
    # torch.save(model.state_dict(), '{train_pos_img_text}.pth')
    import matplotlib
    matplotlib.use('Agg')

    fig, axs = plt.subplots(2, 1, figsize=(10, 15))
  
    # 第一个子图：损失曲线  
    axs[0].plot(train_losses, label='Train Loss')
    axs[0].plot(val_losses, label='Test Loss')
    axs[0].legend()
    axs[0].set_title("Loss Curve")

    
    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"  
                f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"  
                f"Train C Accuracy: {best_epoch_info['train_classification_accuracy']:.4f}\n"  
                f"Val Loss: {best_epoch_info['val_loss']:.4f}\n"  
                f"Val C Accuracy: {best_epoch_info['val_classification_accuracy']:.4f}\n"
                )  

    axs[1].axis('off')
    axs[1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[1].transAxes)

    plt.tight_layout()  # 确保子图之间的间距合适  

    os.makedirs(f"./figure/contrast/{config['encoder_type']}/R+C/", exist_ok=True)
    plotpath = f"./figure/contrast/{config['encoder_type']}/R+C/{current_time}.png"
    plt.suptitle('R+C ' + topo, fontsize=16, y=1.05)
    plt.savefig(plotpath)

    # draw_tsne_eeg(train_eegfeature, train_eeglabel, state='train')
    # draw_tsne_eeg(eegfeature, eeglabel,state='test')
    return results,file_path

import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
def main():
    seed_n = int(os.environ['SEED_N'])
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    np.random.seed(seed_n)
    random.seed(seed_n)
    config = {
        "data_path": f"/home/ubuntu/桌面/xhx/imagenet40-1000-1_eeg-100hz_1s.csv",
        "image_path": f"/home/ubuntu/桌面/xhx/imagenet40-1000-1_img-100hz_1s.csv",
        "project": "train_pos_img_text_rep",
        "n_classes": 40,
        "lr": 3e-4,
        "epochs": 1,
        "batch_size": 64,
        "insubject": True,
        "encoder_type": 'Model',
        "img_encoder": 'Proj_img',
        "topo": 'all',# frontal,central,parietal,temporal,occipital
        'strategy': 'R+C',
        "channel": {"all": 96, "frontal": 26, "central": 18, "parietal": 20, "temporal": 6, "occipital": 26}
    }


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    # Re-initialize the models for each subject

    #config['encoder_type'] = model
    eeg_model = globals()[config['encoder_type']](config['strategy'],num_channels=config['channel'][config['topo']])
    eeg_model.to(device)

    optimizer = torch.optim.AdamW(itertools.chain(eeg_model.parameters()), lr=config['lr'])
    # 定义学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    print(f'Processing number of parameters:',sum([p.numel() for p in itertools.chain(eeg_model.parameters())]))
    all_imgclass, all_imglabel = get_img_features(config['image_path'])
    dataset = CusEEGDataset(config['topo'], config['data_path'], config['image_path'], device)
    dataset_size = len(dataset)

    # 计算每个子集的大小
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = int(0.1 * dataset_size)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    results,file_path = main_train_loop(config['topo'], current_time, eeg_model, train_loader, val_loader, test_loader, optimizer, scheduler, device, all_imgclass,all_imglabel, config)
    #last test
    model_name = globals()[config['encoder_type']]
    eeg_model = load_pretrained_eeg_model(file_path, config['strategy'], model_name, config['channel'])
    test_loss, test_retrieval_accuracy, top3_acc, test_classification_accuracy, eegfeature, eeglabel = evaluate_model(
        eeg_model, test_loader, device, all_imgclass, all_imglabel)
    print(f"Test_retrieval_Accuracy: {test_retrieval_accuracy:.4f},Top3_Acc: {top3_acc:.4f}, Test_classification_Accuracy: {test_classification_accuracy:.4f}")

    # Save results to a CSV file
    results_dir = f"./outputs/contrast/{config['encoder_type']}/{current_time}/{config['encoder_type']}"
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/{config['topo']}_R+C.csv"

    with open(results_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'Results saved to {results_file}')

if __name__ == '__main__':
        main()