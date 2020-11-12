# third party imports
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is 3 x 224 x 224
            nn.Conv2d(3, 32, 3, 1, 0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, 3, 1, 0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2, inplace=True),
            
            nn.Conv2d(128, 32, 1, 1, 0, bias=False),
            
            nn.Conv2d(32, 64, 3, 1, 0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2, inplace=True),
            
            nn.Conv2d(128, 32, 1, 1, 0, bias=False),
            
            nn.Conv2d(32, 64, 3, 1, 0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2, inplace=True),
            
            nn.Conv2d(128, 32, 1, 1, 0, bias=False),
            
            nn.Conv2d(32, 64, 3, 1, 0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2, inplace=True),
            
            nn.Conv2d(128, 64, 1, 1, 0, bias=False),
            
            nn.AvgPool2d(kernel_size = 10),
            nn.Flatten(),
            
            nn.Linear(64, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 1),
            
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)