import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class conv(nn.Module):
    def __init__(self):
        super(conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=5, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        # self.conv = nn.Sequential(
        #     nn.Conv1d(5, 16, kernel_size=5, padding=2),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     nn.Conv1d(16, 32, kernel_size=5, padding=2),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Conv1d(32, 64, kernel_size=5, padding=2),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 128, kernel_size=5, padding=2),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Conv1d(128, 256, kernel_size=5, padding=2),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU()
        # )
        self.lin = nn.Sequential(nn.Linear(11264, 2))
                                 # nn.Linear(320, 2))

    def forward(self, x):
        f1 = x[:,0:64]
        f2 = x[:,64:128]
        f3 = x[:,128:192]
        f4 = x[:,192:256]
        f5 = x[:,256:320]

        x = torch.cat([f1.unsqueeze(dim=1), f2.unsqueeze(dim=1), f3.unsqueeze(dim=1), f4.unsqueeze(dim=1), f5.unsqueeze(dim=1)], dim=1)
        x = self.conv(x)
        x = torch.flatten(x,1)
        return self.lin(x)
        # f1 = self.conv(f1.unsqueeze(dim=1))
        # f1 = torch.flatten(f1,1)
        # f1 = self.lin(f1)
        #
        # f2 = self.conv(f2.unsqueeze(dim=1))
        # f2 = torch.flatten(f2,1)
        # f2 = self.lin(f2)
        #
        # f3 = self.conv(f3.unsqueeze(dim=1))
        # f3 = torch.flatten(f3,1)
        # f3 = self.lin(f3)
        #
        # f4 = self.conv(f4.unsqueeze(dim=1))
        # f4 = torch.flatten(f4,1)
        # f4 = self.lin(f4)
        #
        # f5 = self.conv(f5.unsqueeze(dim=1))
        # f5 = torch.flatten(f5,1)
        # f5 = self.lin(f5)
        #
        #
        # all = torch.cat((f1.unsqueeze(0),f2.unsqueeze(0),f3.unsqueeze(0),f4.unsqueeze(0),f5.unsqueeze(0)))
        # diff = all[:,:,1] - all[:,:,0]
        # max_pos = torch.argmax(diff, dim=0)
        # return all[max_pos, np.arange(all.shape[1])]

