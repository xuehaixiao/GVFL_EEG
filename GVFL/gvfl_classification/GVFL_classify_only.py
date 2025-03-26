import os
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
import matplotlib.pyplot as plt
# from t_sne import draw_tsne,draw_tsne_eeg
import torch.nn as nn
from LoaderCon import CusEEGDataset
import random
import torch
from torch.utils.data import DataLoader,random_split,Subset
import csv
import itertools
import numpy as np
from GVFL.models.EEGNet import EEGNet
from GVFL.models.ATMS_50 import ATMS_50
from GVFL.models.encoder import Model,Proj_img
from GVFL.models.NICE import GA_NICE,SA_NICE,Deepnet_NICE,Shallownet_NICE
from GVFL.models.EEGChannelnet import EEGChannelnet
from GVFL.models.classify_head import ClassifierHead
from marix_draw import  ConfusionMatrix


def train_model(eeg_model,  dataloader, optimizer, device):
    eeg_model.train()
    total_loss = 0
    correct = 0
    total = 0

    criteon = nn.CrossEntropyLoss()

    for batch_idx, (eeg_data, labels, _) in enumerate(dataloader):
       
        eeg_data = eeg_data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        out = eeg_model(eeg_data)

        loss = criteon(out.float(), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        #总数
        batch_size = labels.shape[0]
        total += batch_size
        #准确个数
        _, pred = out.data.max(1)
        correct += pred.eq(labels.data).sum().item()
        
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    return average_loss, accuracy

def evaluate_model(eeg_model, dataloader, device):
    eeg_model.eval()

    criteon = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    labels = [i for i in range(40)]
    confusion = ConfusionMatrix(num_classes=40, labels=labels, normalize=True, batch_size=64)
    
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, _) in enumerate(dataloader):
            
            eeg_data = eeg_data.to(device)
            labels = labels.to(device)
            
            out = eeg_model(eeg_data)
            loss = criteon(out.float(), labels)
            total_loss += loss.item()
            eeg_class_marix = torch.softmax(out, dim=1)
            eeg_class_marix = torch.argmax(eeg_class_marix, dim=1)
            confusion.update(eeg_class_marix.to("cpu").numpy(), labels.to("cpu").numpy())

            #总数
            batch_size = labels.shape[0]
            total += batch_size
            #准确个数
            _, pred = out.data.max(1)
            correct += pred.eq(labels.data).sum().item()
                    
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    metric = confusion.summary()
    return average_loss, accuracy,metric
def main_train_loop(topo,current_time, eeg_model, train_dataloader,val_dataloader, test_dataloader, optimizer, device,
             config,epochs):

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    

    best_accuracy = 0.0
    best_model_weights = None
    best_epoch_info = {}
    results = []  # List to store results for each epoch
    
    for epoch in range(epochs):
        
        train_loss, train_accuracy = train_model(eeg_model, train_dataloader, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)


        
        val_loss, val_accuracy,_ = evaluate_model(eeg_model, val_dataloader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Append results for this epoch
        epoch_results = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        }

        results.append(epoch_results)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_weights = eeg_model.state_dict().copy()
            
            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    eeg_model.load_state_dict(best_model_weights)
    test_loss, test_accuracy,metric = evaluate_model(eeg_model, test_dataloader, device)
    print(f"Finished!Test_classification_Accuracy: {test_accuracy:.4f}")
    # torch.save(model.state_dict(), '{train_pos_img_text}.pth')
    import matplotlib
    matplotlib.use('Agg')

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))  # 注意，figsize 的高度可能不需要 15 这么高，因为只有一个行

    # 第一个子图：损失曲线
    axs[0].plot(train_losses, label='Train Loss')
    axs[0].plot(val_losses, label='Test Loss')
    axs[0].legend()
    axs[0].set_title("Loss Curve")

    # 第二个子图：准确率曲线
    axs[1].plot(train_accuracies, label='Train Accuracy')
    axs[1].plot(val_accuracies, label='Test Accuracy')
    axs[1].legend()
    axs[1].set_title("Accuracy Curve")



    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
                f"Val Loss: {best_epoch_info['val_loss']:.4f}\n"
                f"Val Accuracy: {best_epoch_info['val_accuracy']:.4f}\n"
                f"Metric - Acc: {test_accuracy:.4f}\n"
                f"Metric - F1: {metric['F1 Score']:.4f}\n"
                f"Metric - Pre: {metric['Precision']:.4f}\n"
                f"Metric - Rec: {metric['Recall']:.4f}\n")

    axs[2].axis('off')
    axs[2].text(0.5, 0.5,  info_text, fontsize=8, transform=axs[2].transAxes)

    plt.tight_layout()

    plotdir = f"./figure/{config['encoder_type']}/classify_only"
    os.makedirs(plotdir, exist_ok=True)
    plotpath = f"{plotdir}/{current_time}_{topo}.png"
    plt.savefig(plotpath)
    return results


import datetime
import json

def main():
    seed_n = np.random.randint(2023)
    print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)

    config_file_path = "/home/ubuntu/桌面/xhx/GVFL/gvfl_classification/config.json"

    with open(config_file_path, 'r') as f:
        config = json.load(f)

    epochs = 150


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")  


    eeg_model = globals()[config['encoder_type']]('classify',num_channels=config['channel'][config['topo']])

    eeg_model.to(device)
    optimizer = torch.optim.AdamW(itertools.chain(eeg_model.parameters()), lr=1e-5)


    print(f'Processing number of parameters:', sum([p.numel() for p in itertools.chain(eeg_model.parameters())]))

    dataset = CusEEGDataset(config['topo'], config['data_path'], config['image_path'], device)
    dataset_size = len(dataset)
    # 计算每个子集的大小
    # train_size = int(0.8 * dataset_size)
    # val_size = int(0.1 * dataset_size)
    # test_size = int(0.1 * dataset_size)
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size,val_size, test_size])
    train_dataset = Subset(dataset, indices=range(32000))
    val_dataset = Subset(dataset, indices=range(32000, 32000 + 4000))
    test_dataset = Subset(dataset, indices=range(32000 + 4000, 40000))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    results = main_train_loop(config['topo'], current_time, eeg_model, train_loader, val_loader, test_loader, optimizer,
                              device,  config, epochs)

    # Save results to a CSV file
    results_dir = f"./outputs/{config['encoder_type']}/classify_only"
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/{config['topo']}_{current_time}.csv"

    with open(results_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'Results saved to {results_file}')
            
if __name__ == '__main__':
    main()