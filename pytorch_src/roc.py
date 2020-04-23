from __future__ import print_function

import argparse
import os
import time
import torch
import torch.nn.parallel
import numpy as np
import torch.utils.data as data
from sklearn import metrics

from models.fcc import FCC
from dataset.nlst import datasetNLST as dataset
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ranksums


parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
parser.add_argument('--resume', default='result_mixmatchnst/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')



args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()


def main():
    global best_acc



    # Model
    print("==> creating WRN-28-2")

    def create_model(ema=False):
        model = FCC()
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model



    # test_these = ['result_standard/model_best.pth.tar',
    #               'result_nst/model_best.pth.tar',
    #               'result_mixmatch/model_best.pth.tar',
    #               'result_mixmatchnst/model_best.pth.tar']
    # test_these = ['result_standard_4110/model_best.pth.tar',
    #               'result_standard_242/model_best.pth.tar',
    #               'result_standard_5/model_best.pth.tar',
    #               'result_standard_1/model_best.pth.tar']
    test_these = ['result_standard_5/model_best.pth.tar',
                  'result_mixmatch_5/model_best.pth.tar',
                  'result_mixmatchnst_5/model_best.pth.tar']
    # legend = ['Standard 4366', 'Standard 242', 'Standard 5', 'Standard 1']
    legend = ['Standard 5', 'MixMatch 5', 'NST 5']
    test_set = dataset('/nfs/masi/hansencb/nlst_nst_mixmatch/data_organization/test.csv', labeled=True, train=False, use_latest=True)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = create_model()

    plt.figure()
    ranky_bois = []
    for i,chckpt_file in enumerate(test_these):

        # Load checkpoint.
        print('==> Loading Model..')
        assert os.path.isfile(chckpt_file), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(chckpt_file)
        checkpoint = torch.load(chckpt_file)
        model.load_state_dict(checkpoint['state_dict'])

        auc, fpr, tpr, scores = validate(test_loader, model, True)

        print('{}: {}'.format(legend[i], auc))
        plt.plot(fpr, tpr)

        ranky_bois.append(scores)

    wc = wilcoxon(ranky_bois[1], ranky_bois[2])
    print(wc)
    plt.legend(legend)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid(True)
    plt.title('NLST Cancer Classification')
    plt.show()

def validate(valloader, model, use_cuda):
    # switch to evaluate mode
    model.eval()
    scores = np.array([])
    y = np.array([])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)
            logits = torch.softmax(outputs, dim=1)[:,1]

            scores = np.concatenate((scores, logits.cpu().numpy()))
            y = np.concatenate((y, targets.cpu().numpy()))

    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    auc = metrics.roc_auc_score(y, scores)
    return auc, fpr, tpr, scores


if __name__ == '__main__':
    main()
