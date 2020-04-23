import numpy as np
from PIL import Image
import random
import torchvision
import torch
from torch.utils.data import Dataset

class datasetNLST(Dataset):
    def __init__(self, data_file, labeled=True, train=True, use_latest=False):
        self.labeled = labeled
        self.train = train
        self.use_latest = use_latest
        if self.train:
            self.data = {}
        else:
            self.data = []
        self.len = 0
        self.scores = []
        with open(data_file, 'r') as f:
            for line in f.readlines():
                paths, label = line.split(',')
                label = int(label)
                if not labeled:
                    label = -1
                paths = paths.split(';')

                if self.train:
                    imgs = []
                    for path in paths:
                        imgs.append(np.load(path))

                        self.len += 1

                    if label not in self.data:
                        self.data[label] = []
                    self.data[label].append((imgs, label))

                    if not labeled:
                        self.scores.append((0, len(self.data[label])-1))
                else:
                    if self.use_latest:
                        latest = 0
                        latest_path = ''
                        for path in paths:
                            time = int(path.split('time')[-1].split('.')[0])
                            if time > latest:
                                latest = time
                                latest_path = path
                        self.data.append((np.load(latest_path), label))
                        self.len+=1
                    else:
                        for path in paths:
                            self.data.append((np.load(path), label))
                            self.len += 1



    def update_scores(self, model, device):
        model.eval()
        # tmp = {0:[], 1:[]}
        with torch.no_grad():
            for label in self.data:
                for i, subj in enumerate(self.data[label]):
                    x = torch.tensor([])
                    for img in subj[0]:
                        img = img.copy()
                        img = torch.from_numpy(img)
                        img = img.view(-1)
                        x = torch.cat((x, img.unsqueeze(0)))

                    x = x.to(device)
                    y = model(x)
                    y = torch.softmax(y, dim = 1)

                    score = torch.max(y[:,1])
                    # c = torch.argmax(y[0,:])
                    # tmp[c.item()].append(self.data[label][i])
                    self.scores[i] = (score.item(), self.scores[i][1])
        self.scores.sort(reverse=True)
        # self.data = tmp



    def __len__(self):
        if self.train:
            return max([self.len, 128])
        else:
            return self.len

    def GaussianNoise(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.2
        return x

    def process(self, img):
        img = np.expand_dims(img, 0)

        if self.train:
            img1 = self.GaussianNoise(img.copy())
            img1 = torch.from_numpy(img1)
            img1 = img1.view(-1)

            if self.labeled:
                return img1

            img2 = self.GaussianNoise(img.copy())
            img2 = torch.from_numpy(img2)
            img2 = img2.view(-1)

            return img1, img2
        else:
            img = torch.from_numpy(img)
            img = img.view(-1)
            return img


    def __getitem__(self, i):
        if self.train:
            target = random.choice(list(self.data.keys()))
            if self.labeled:
                subjidx = random.randint(0, len(self.data[target])-1)
            else:
                # subjidx = random.randint(0, len(self.data[target]) - 1)
                # # if random.randint(0,1) == 0:
                scoreidx = random.randint(0, int((len(self.scores)-1)*0.25))
                # # else:
                # #     scoreidx = random.randint(int((len(self.scores) - 1) * 0.25), len(self.scores) - 1)
                subjidx = self.scores[scoreidx][1]

            subj = self.data[target][subjidx]

            imgidx = random.randint(0, len(subj[0])-1)

            img = subj[0][imgidx]
            img = self.process(img)
        else:
            target = self.data[i][1]
            img = self.data[i][0]
            img = self.process(img)
        if self.labeled:
            return img, torch.tensor(target)
        else:
            target = -1
            equiv_img = img

            if len(subj[0]) > 1:
                eimgidx = random.choice([x for x in range(0, len(subj[0])) if x != imgidx])
                equiv_img = subj[0][eimgidx]
                equiv_img = self.process(equiv_img)

            return img, equiv_img, target

