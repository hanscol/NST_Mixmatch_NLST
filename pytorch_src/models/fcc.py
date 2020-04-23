import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# class FCC(nn.Module):
#     def __init__(self):
#         super(FCC, self).__init__()
#
#         self.lin1 = nn.Linear(320, 320)
#         self.lin2 = nn.Linear(320, 256)
#         self.lin3 = nn.Linear(256, 256)
#         self.lin4 = nn.Linear(256, 128)
#         self.lin5 = nn.Linear(128, 128)
#         self.lin6 = nn.Linear(128, 64)
#         self.lin7 = nn.Linear(64, 64)
#         self.lin8 = nn.Linear(64, 16)
#         self.lin9 = nn.Linear(16, 16)
#         self.lin10 = nn.Linear(16, 2)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.relu(self.lin1(x))
#         x = self.relu(self.lin2(x))
#         x = self.relu(self.lin3(x))
#         x = self.relu(self.lin4(x))
#         x = self.relu(self.lin5(x))
#         x = self.relu(self.lin6(x))
#         x = self.relu(self.lin7(x))
#         x = self.relu(self.lin8(x))
#         x = self.relu(self.lin9(x))
#         x = self.lin10(x)
#         return x

class FCC(nn.Module):
    def __init__(self):
        super(FCC, self).__init__()

        # self.lin = nn.Sequential(nn.Linear(64, 128), nn.Linear(128, 256), nn.Linear(256, 128), nn.Linear(128, 64), nn.Linear(64, 2))
        # self.lin = nn.Linear(64,2)
        # self.lin = nn.Sequential(nn.Linear(64, 64),
        #                          nn.Linear(64, 64),
        #                          nn.Linear(64, 64),
        #                          nn.Linear(64, 64),
        #                          nn.Linear(64, 64),
        #                          nn.Linear(64, 64),
        #                          nn.Linear(64, 64),
        #                          nn.Linear(64, 2))
        self.lin = nn.Sequential(nn.Linear(320, 320),
                                 nn.Linear(320, 320),
                                 nn.Linear(320, 320),
                                 nn.Linear(320, 2))
        # self.lin = nn.Sequential(nn.Linear(320, 320), nn.Linear(320, 2))

    def forward(self, x):
        return self.lin(x)

        # f1 = self.lin(x[:,0:64])
        # f2 = self.lin(x[:,64:128])
        # f3 = self.lin(x[:,128:192])
        # f4 = self.lin(x[:,192:256])
        # f5 = self.lin(x[:,256:320])
        #
        # all = torch.cat((f1.unsqueeze(0),f2.unsqueeze(0),f3.unsqueeze(0),f4.unsqueeze(0),f5.unsqueeze(0)))
        # diff = all[:,:,1] - all[:,:,0]
        # max_pos = torch.argmax(diff, dim=0)
        # return all[max_pos, np.arange(all.shape[1])]

        # x1 = x[:,0:64]
        # x2 = x[:,64:128]
        # x3 = x[:,128:192]
        # x4 = x[:,192:256]
        # x5 = x[:,256:320]
        # l = [x1, x2, x3, x4, x5]
        #
        #
        # f1 = self.lin(torch.cat(l, dim=1))
        # l = [x2, x3, x4, x5, x1]
        # f2 = self.lin(torch.cat(l, dim=1))
        # l = [x3, x4, x5, x1, x2]
        # f3 = self.lin(torch.cat(l, dim=1))
        # l = [x4, x5, x1, x2, x3]
        # f4 = self.lin(torch.cat(l, dim=1))
        # l = [x5, x1, x2, x3, x4]
        # f5 = self.lin(torch.cat(l, dim=1))
        #
        # return (f1+f2+f3+f4+f5)/5

        # all = torch.cat((f1.unsqueeze(0),f2.unsqueeze(0),f3.unsqueeze(0),f4.unsqueeze(0),f5.unsqueeze(0)))
        # diff = all[:,:,1] - all[:,:,0]
        # max_pos = torch.argmax(diff, dim=0)
        # return all[max_pos, np.arange(all.shape[1])]
