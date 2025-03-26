import torch.nn as nn
class ClassifierHead(nn.Sequential):
    def __init__(self, ):
        super().__init__()
        self.l=nn.Linear(1024, 40)
    def forward(self,x):
        x = self.l(x)
        return x