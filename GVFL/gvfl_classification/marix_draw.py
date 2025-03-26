import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from prettytable import PrettyTable
import itertools
import pandas as pd




class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    def __init__(self, num_classes: int, labels: list, normalize: bool, batch_size: int):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.normalize = normalize
        self.batch_size = batch_size

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table1 = PrettyTable()
        table2 = PrettyTable()
        table1.field_names = [" ","Precision", "Recall", "Specificity","F1 Score"]
        f1_scores=[]
        recall_scores=[]
        specificity_scores=[]
        precision_scores=[]
        overall_TP = overall_FP = overall_FN = overall_TN = 0
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            overall_TP += TP
            overall_FP += FP
            overall_FN += FN
            overall_TN += TN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1 = round(2*(Precision*Recall)/(Precision+Recall),3)if Precision*Recall!=0 else 0.
            recall_scores.append(Recall)
            f1_scores.append(F1)
            specificity_scores.append(Specificity)
            precision_scores.append(Precision)
            table1.add_row([self.labels[i], Precision, Recall, Specificity,F1])
        macro_f1 = np.mean(f1_scores)
        overall_Recall = np.mean(recall_scores)
        overall_Specificity = np.mean(specificity_scores)
        overall_Precision = np.mean(precision_scores)

        table2.field_names = ["Overall", "Precision", "Recall", "Specificity", "F1 Score", "Micro"]
        table2.add_row(["Overall", overall_Precision, overall_Recall, overall_Specificity, round(macro_f1, 3), acc])
        metrics_dict = {
            "Overall": "Overall",
            "Precision": overall_Precision,
            "Recall": overall_Recall,
            "Specificity": overall_Specificity,
            "F1 Score": round(macro_f1, 3),
            "Micro": acc
        }

        #print(table1)
        print(table2)
        return metrics_dict


    def plot(self):
        self.plot_confusion_matrix()

    def plot_confusion_matrix(self):
        # 创建一个DataFrame来保存混淆矩阵
        filename='/home/ubuntu/桌面/xhx/EEG40000/marix.xlsx'
        df_cm = pd.DataFrame(self.matrix, index=self.labels, columns=self.labels)
        if filename:
            # 检查文件扩展名来决定保存格式
            if filename.endswith('.csv'):
                df_cm.to_csv(filename, index=True)
            elif filename.endswith('.xlsx'):
                df_cm.to_excel(filename, index=True)
            else:
                raise ValueError("Unsupported file format. Use .csv or .xlsx.")

        matrix = self.matrix
        classes = self.labels
        normalize = self.normalize
        title = 'Confusion matrix'

        print('normalize: ', normalize)

        if normalize:
            matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
            print("显示百分比：")
            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
            print(matrix)
        else:
            print('显示具体数字：')
            print(matrix)

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=classes,
            y=classes,
            hoverongaps=False,
            colorscale='Blues',
            showscale=True
        ))

        fig.update_layout(
            title=title,
            xaxis=dict(title='Predicted label'),
            yaxis=dict(title='True label'),
            yaxis_autorange='reversed'  # Ensures the y-axis labels are displayed in the correct order
        )

        #fig.show()




