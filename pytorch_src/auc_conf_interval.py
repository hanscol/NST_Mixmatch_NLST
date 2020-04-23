import numpy as np
import os
import stat_util
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import sklearn.metrics
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def main():
    y_true = []
    for i in range(106):
        test_file = '/nfs/masi/hansencb/nlst_nst_mixmatch/data_organization/fold{}/test.csv'.format(i)
        tmp = []
        with open(test_file, 'r') as f:
            for line in f.readlines():
                tmp.append(int(line.strip().split(',')[-1]))
        y_true.append(tmp)

    r = robjects.r
    r['source']('/nfs/masi/hansencb/nlst_nst_mixmatch/ROCfunctions/ROCplot.R')
    r['source']('/nfs/masi/hansencb/nlst_nst_mixmatch/ROCfunctions/ROC.R')
    r['source']('/nfs/masi/hansencb/nlst_nst_mixmatch/ROCfunctions/AUC.R')
    rocplotfun = r['ROC.plot']
    rocfun = r['ROC']
    aucfun = r['AUC']

    data = {'standard':{}, 'nst':{}, 'mix':{}, 'nstmix':{}}
    scores = {'standard':{}, 'nst':{}, 'mix':{}, 'nstmix':{}}
    results_dir = '/nfs/masi/hansencb/nlst_nst_mixmatch/results'
    result_dirs = os.listdir(results_dir)
    result_dirs.sort()
    # nums = [4, 20, 40, 200, 400, 800, 1200, 2000, 2800, 3600]
    nums = [40, 200, 400]


    for result_dir in result_dirs:
        parts = result_dir.split('_')
        # key = '_'.join(parts[1:])

        fold = int(parts[0][-1])
        num = int(parts[1].split('d')[-1])
        nst = float(parts[2].split('a')[-1])
        lamb = float(parts[3].split('a')[-1])

        for n in nums:
            if abs(num-n) < 5:
                num = n

        key = '{}_{}_{}'.format(num, nst, lamb)

        npz_file = os.path.join(results_dir, result_dir, 'metrics.npz')
        result_path = os.path.join(results_dir, result_dir)

        if os.path.isfile(npz_file):
            npz = np.load(npz_file)
            auc = npz['arr_0']
            fpr = npz['arr_1']
            tpr = npz['arr_2']
            score = npz['arr_3']
            aucvals = aucfun(robjects.FloatVector(score), robjects.FloatVector(y_true[fold]))
            auc = np.array(aucvals[0])
            ci = np.array(aucvals[1])

            if nst == 0 and lamb == 0:
                if num not in data['standard']:
                    data['standard'][num] = [[], []]
                data['standard'][num][0].append(auc)
                data['standard'][num][1].append(ci)

            elif nst != 0 and lamb != 0:
                key = '{}_{}'.format(nst, lamb)
                if num not in data['nstmix']:
                    data['nstmix'][num] = {}
                if key not in data['nstmix'][num]:
                    data['nstmix'][num][key] = [[], []]
                data['nstmix'][num][key][0].append(auc)
                data['nstmix'][num][key][1].append(ci)

            elif nst != 0:
                if num not in data['nst']:
                    data['nst'][num] = {}
                if nst not in data['nst'][num]:
                    data['nst'][num][nst] = [[], []]
                data['nst'][num][nst][0].append(auc)
                data['nst'][num][nst][1].append(ci)

            else:
                if num not in data['mix']:
                    data['mix'][num] = {}
                if lamb not in data['mix'][num]:
                    data['mix'][num][lamb] = [[], []]
                data['mix'][num][lamb][0].append(auc)
                data['mix'][num][lamb][1].append(ci)

    # nums = nums[1:-3]

    img = []
    for num in nums:
        d = data['nst'][num]
        nsts = list(d.keys())
        nsts.sort()

        row = []
        for nst in nsts:
            row.append(np.mean(d[nst][0]))
        img.append(row)

        data['nst'][num] = data['nst'][num][nsts[np.argmax(row)]]


    img = np.array(img)

    img = []
    for num in nums:
        d = data['mix'][num]
        lambs = list(d.keys())
        lambs.sort()

        row = []
        for lamb in lambs:
            row.append(np.mean(d[lamb][0]))
        img.append(row)
        data['mix'][num] = data['mix'][num][lambs[np.argmax(row)]]

    img = []
    for num in nums:
        d = data['nstmix'][num]
        hypers = list(d.keys())

        row = []
        for hyper in hypers:
            if len(d[hyper][0])>1:
                row.append(np.mean(d[hyper][0]))
            else:
                row.append(0)
        img.append(row)
        data['nstmix'][num] = data['nstmix'][num][hypers[np.argmax(row)]]


    combos = [[200, 0], [200, 1], [200, 2], [200, 3], [200, 4]]
    # nums = [40, 200, 400, 800]
    fig, ax = plt.subplots(2,3)
    legend = ['Standard', 'Nullspace Tuning', 'MixMatch', 'MixMatchNST', 'Baseline']
    axx = 0
    axy = 0
    for i in range(5):
        for method in scores:
            y = []
            ytop = []
            ybot = []
            for num in nums:
                y.append(data[method][num][0][i])
                ytop.append(data[method][num][1][i][1])
                ybot.append(data[method][num][1][i][0])

            x = np.arange(0, len(nums))
            ax[axx, axy].fill_between(x, ybot, ytop, alpha=0.5)
            ax[axx, axy].plot(x,y)

        orig_auc = 0.7418
        x = np.arange(0, len(nums))
        ax[axx,axy].plot(x, np.zeros(len(nums)) + orig_auc)
        ax[axx,axy].fill_between(x, np.zeros(len(nums)) + orig_auc + 0, np.zeros(len(nums)) + orig_auc - 0, alpha=0.25)

        ax[axx,axy].legend(legend)
        ax[axx,axy].grid(True)
        ax[axx,axy].set_xticks(np.arange(0,len(nums)))
        ax[axx,axy].set_xticklabels(nums)
        ax[axx,axy].set_xlabel('Num labeled Subjects')
        ax[axx,axy].set_ylabel('AUC')

        axy += 1
        if axy == 3:
            axx+=1
            axy = 0
    plt.show()


    # patches = []
    # legend = list(exp_scores.keys())
    # fig, ax = plt.subplots()
    # for method in exp_scores:
    #     plotdata = rocplotfun(robjects.FloatVector(exp_scores[method]), robjects.FloatVector(y_true))
    #     fp = np.array(plotdata[0])
    #     tp = np.array(plotdata[1])
    #     regionx = np.array(plotdata[2])
    #     regiony = np.array(plotdata[3])
    #     poly = np.stack((regionx, regiony), 1)
    #     patches.append(Polygon(poly))
    #     ax.plot(fp, tp)
    #
    # p = PatchCollection(patches, cmap='tab20', alpha=0.5)
    # p.set(array=np.array([0, 1, 2, 3, 4]))
    # ax.add_collection(p)
    # ax.legend(legend)
    # plt.show()


if __name__ == '__main__':
    main()