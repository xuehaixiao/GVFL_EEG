import os
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
import matplotlib.pyplot as plt
# from t_sne import draw_tsne,draw_tsne_eeg
import torch.nn as nn
from LoaderCon import CusEEGDataset
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

def train_model(eeg_model, classifier_head, dataloader, optimizer, device):
    eeg_model.train()
    classifier_head.train()
    total_loss = 0
    correct_classification = 0
    total = 0
    criteon = nn.CrossEntropyLoss()
    eeg_model = eeg_model.cuda()

    for batch_idx, (eeg_data, labels, _) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        # eeg_data = torch.rand(eeg_data.size())  # .to(device)
        # mean = eeg_data.mean()
        # std = eeg_data.std()
        # eeg_data = ((eeg_data - mean) / std).to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        eeg_features = eeg_model(eeg_data)
        eeg_class = classifier_head(eeg_features).float()
        
        loss2 = criteon(eeg_class,labels)
        loss = loss2
        loss.backward()
        optimizer.step()
        

        total_loss += loss.item()
        
 
        batch_size = labels.shape[0]
        total += batch_size
        #准确个数
        _, pred = eeg_class.data.max(1)
        correct_classification += pred.eq(labels.data).sum().item()
        
    average_loss = total_loss / (batch_idx+1)
    correct_classification = correct_classification / total
    return average_loss,correct_classification 
def evaluate_model(eeg_model, classifier_head, dataloader, device):
    eeg_model.eval()
    classifier_head.eval()
    total_loss = 0
    correct_classification = 0
    total = 0

    criteon = nn.CrossEntropyLoss()
    labels = [i for i in range(40)]
    confusion = ConfusionMatrix(num_classes=40, labels=labels, normalize=True, batch_size=64)
    eeg_model = eeg_model.cuda()
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, _) in enumerate(dataloader):
            
            eeg_data = eeg_data.to(device)
            # eeg_data = torch.rand(eeg_data.size())#.to(device)
            # mean = eeg_data.mean()
            # std = eeg_data.stdeeg_data = torch.rand(eeg_data.size())#.to(device)

            # eeg_data = ((eeg_data - mean) / std).to(device)

            labels = labels.to(device)
            
            eeg_features = eeg_model(eeg_data)
            eeg_class = classifier_head(eeg_features).float()
            eeg_class_marix = torch.softmax(eeg_class, dim=1)
            eeg_class_marix = torch.argmax(eeg_class_marix, dim=1)
            confusion.update(eeg_class_marix.to("cpu").numpy(), labels.to("cpu").numpy())

            loss2 = criteon(eeg_class, labels)
            loss =loss2
            total_loss += loss.item()
        
            batch_size = labels.shape[0]
            total += batch_size
            #准确个数
            _, pred = eeg_class.data.max(1)
            correct_classification += pred.eq(labels.data).sum().item()
        
    average_loss = total_loss / (batch_idx+1)
    correct_classification = correct_classification / total
    metric = confusion.summary()
    return average_loss,correct_classification,metric

def main_train_loop(topo, current_time, eeg_model, classifier_head, train_dataloader, val_dataloader, test_dataloader, optimizer, device,
             config,epochs):

    train_losses, train_retrieval_accuracies, train_classification_accuracies = [], [], []
    #test_losses, test_retrieval_accuracies, top3_accs, test_classification_accuracies = [], [], [], []
    val_losses, val_retrieval_accuracies, top3_accs, val_classification_accuracies = [], [], [], []

    best_accuracy = 100
    best_model_weights_cls = None
    best_model_weights_eeg = None

    best_epoch_info = {}
    results = []  # List to store results for each epoch
    
    for epoch in range(epochs):
        #traing
        train_loss, train_classification_accuracy = train_model(eeg_model,classifier_head, train_dataloader, optimizer, device)
        train_losses.append(train_loss)
        train_classification_accuracies.append(train_classification_accuracy)
        #eval
        val_loss, val_classification_accuracy,val_metric = evaluate_model(eeg_model, classifier_head, val_dataloader, device)
        val_losses.append(val_loss)
        val_classification_accuracies.append(val_classification_accuracy)

        #save best model
        if best_accuracy > val_loss:
            best_epoch = epoch
            best_accuracy = val_loss
            best_model_weights_eeg = eeg_model.state_dict()
            best_model_weights_cls = classifier_head.state_dict()
            print('best_epoch:', best_epoch)



        # Append results for this epoch
        epoch_results = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_classification_accuracy": train_classification_accuracy,
        "val_loss": val_loss,
        "val_classification_accuracy": val_classification_accuracy,
        }

        results.append(epoch_results)

            
        best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_classification_accuracy": train_classification_accuracy,
                "val_loss": val_loss,
                "val_classification_accuracy": val_classification_accuracy,
                "F1":val_metric['F1 Score'],
                "Pre":val_metric['Precision'],
                "Rec": val_metric['Recall'],
            }


        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train_classification_Accuracy: {train_classification_accuracy:.4f}")
        print(f"Epoch {epoch + 1}/{epochs} - Val Loss: {val_loss:.4f}, Val_classification_Accuracy: {val_classification_accuracy:.4f}")
    # eeg_model.apply(weights_init_normal)
    # classifier_head.apply(weights_init_normal)
    # file_path_eeg = f"/home/ubuntu/桌面/xhx/EEG40000/models/pretrain/eeg_{current_time}.pth"
    # torch.save(best_model_weights_eeg, file_path_eeg)
    # file_path_cls = f"/home/ubuntu/桌面/xhx/EEG40000/models/pretrain/cls_{current_time}.pth"
    # torch.save(best_model_weights_cls, file_path_cls)
    eeg_model.load_state_dict(best_model_weights_eeg)
    classifier_head.load_state_dict(best_model_weights_cls)

    _, test_classification_accuracy,metric = evaluate_model(eeg_model, classifier_head, test_dataloader, device)

    print(f"Test_classification_Accuracy: {test_classification_accuracy:.4f}")

    # torch.save(model.state_dict(), '{train_pos_img_text}.pth')
    import matplotlib
    matplotlib.use('Agg')

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
  
    # 第一个子图：损失曲线  
    axs[0].plot(train_losses, label='Train Loss')  
    axs[0].plot(val_losses, label='Val Loss')
    axs[0].legend()  
    axs[0].set_title("Loss Curve")  
      
    
    # 第er个子图：分类准确率曲线  
    axs[1].plot(train_classification_accuracies, label='Train Classification Accuracy')
    axs[1].plot(val_classification_accuracies, label='Val Classification Accuracy')
    axs[1].legend()  
    axs[1].set_title("Classification Accuracy Curve")

    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                 f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                 f"Train Accuracy: {best_epoch_info['train_classification_accuracy']:.4f}\n"
                 f"Val Loss: {best_epoch_info['val_loss']:.4f}\n"
                 f"Val Accuracy: {best_epoch_info['val_classification_accuracy']:.4f}\n"
                 f"val_Metric - F1: {best_epoch_info['F1']:.4f}\n"
                 f"val_Metric - Pre: {best_epoch_info['Pre']:.4f}\n"
                 f"val_Metric - Rec: {best_epoch_info['Rec']:.4f}\n"
                 f"Metric - Acc: {test_classification_accuracy:.4f}\n"
                 f"Metric - F1: {metric['F1 Score']:.4f}\n"
                 f"Metric - Pre: {metric['Precision']:.4f}\n"
                 f"Metric - Rec: {metric['Recall']:.4f}\n")
    
    axs[2].axis('off')
    axs[2].text(0.5, 0.5, info_text, fontsize=8, transform=axs[2].transAxes)
    
    plt.tight_layout()  # 确保子图之间的间距合适  

    plotdir = f"./figure/{config['encoder_type']}/classify"
    os.makedirs(plotdir, exist_ok=True)
    plotpath = f"{plotdir}/tm:{config['time']}_topo:{config['topo']}_hz:{config['frequency']}_{current_time}.png"
    plt.savefig(plotpath)


    return results

def load_pretrained_eeg_model(path,strategy,model,channel):
    # 假设模型是一个包含state_dict的字典  
    pretrained_weights = torch.load(path)
    #print(checkpoint.keys())
    eeg_model = model(strategy,num_channels=channel)  # 假设这是您的EEG模型类
    eeg_model.load_state_dict(pretrained_weights)
    return eeg_model  
import json
import datetime
#from GVFL.data_preprocess.bin_self.meanclass import meanclass
def main():

    config_file_path = "/home/ubuntu/桌面/xhx/GVFL/gvfl_classification/config.json"

    with open(config_file_path, 'r') as f:
        config = json.load(f)


    seed_n= config['seed']
    print('seed is ' + str(seed_n))
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    np.random.seed(seed_n)
    random.seed(seed_n)



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # lr = 0.00001
    epochs = 20
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    eeg_model = load_pretrained_eeg_model(config['pretrain_model'], config['strategy'], globals()[config['encoder_type']], config['channel'][config['topo']])
    classifier_head = ClassifierHead()

    eeg_model.to(device)
    classifier_head.to(device)
    optimizer = torch.optim.AdamW(itertools.chain(eeg_model.parameters()), lr=1e-5)

    print(f'Processing number of parameters:', sum([p.numel() for p in itertools.chain(eeg_model.parameters())]))


    dataset = CusEEGDataset(config['topo'], config['data_path'], config['image_path'],device)
    dataset_size = len(dataset)
    # 计算每个子集的大小
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = int(0.1 * dataset_size)
    train_dataset,val_dataset, test_dataset = random_split(dataset, [train_size,val_size, test_size])


    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    # train_loader, val_loader, test_loader = meanclass()
    results = main_train_loop(config['topo'], current_time, eeg_model,classifier_head, train_loader,val_loader, test_loader, optimizer, device,
                                config,epochs)
    # Save results to a CSV file
    results_dir = f"./outputs/{config['encoder_type']}/pretrain"
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/{config['topo']}_hz:{config['frequency']}_{current_time}.csv"

    with open(results_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'Results saved to {results_file}')

if __name__ == '__main__':
    main()