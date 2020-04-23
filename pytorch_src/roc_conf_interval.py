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
    for i in range(5):
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
    nums = [4, 20, 40, 200, 400, 800, 1200, 2000, 2800, 3600]


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

            if nst == 0 and lamb == 0:
                if num not in data['standard']:
                    data['standard'][num] = []
                    scores['standard'][num] = []
                data['standard'][num].append(auc)
                scores['standard'][num].append(score.copy())
            elif nst != 0 and lamb != 0:
                key = '{}_{}'.format(nst, lamb)
                if num not in data['nstmix']:
                    data['nstmix'][num] = {}
                    scores['nstmix'][num] = {}
                if key not in data['nstmix'][num]:
                    data['nstmix'][num][key] = []
                    scores['nstmix'][num][key] = []
                data['nstmix'][num][key].append(auc)
                scores['nstmix'][num][key].append(score.copy())
            elif nst != 0:
                if num not in data['nst']:
                    data['nst'][num] = {}
                    scores['nst'][num] = {}
                if nst not in data['nst'][num]:
                    data['nst'][num][nst] = []
                    scores['nst'][num][nst] = []
                data['nst'][num][nst].append(auc)
                scores['nst'][num][nst].append(score.copy())
            else:
                if num not in data['mix']:
                    data['mix'][num] = {}
                    scores['mix'][num] = {}
                if lamb not in data['mix'][num]:
                    data['mix'][num][lamb] = []
                    scores['mix'][num][lamb] = []
                data['mix'][num][lamb].append(auc)
                scores['mix'][num][lamb].append(score.copy())

    nums = nums[1:-3]

    img = []
    for num in nums:
        d = data['nst'][num]
        nsts = list(d.keys())
        nsts.sort()

        row = []
        for nst in nsts:
            row.append(np.mean(d[nst]))
        img.append(row)

        data['nst'][num] = data['nst'][num][nsts[np.argmax(row)]]
        scores['nst'][num] = scores['nst'][num][nsts[np.argmax(row)]]

    img = np.array(img)

    img = []
    for num in nums:
        d = data['mix'][num]
        lambs = list(d.keys())
        lambs.sort()

        row = []
        for lamb in lambs:
            row.append(np.mean(d[lamb]))
        img.append(row)
        data['mix'][num] = data['mix'][num][lambs[np.argmax(row)]]
        scores['mix'][num] = scores['mix'][num][lambs[np.argmax(row)]]

    img = np.array(img)



    img = []
    for num in nums:
        d = data['nstmix'][num]
        hypers = list(d.keys())

        row = []
        for hyper in hypers:
            if len(d[hyper])>1:
                row.append(np.mean(d[hyper]))
            else:
                row.append(0)
        img.append(row)
        data['nstmix'][num] = data['nstmix'][num][hypers[np.argmax(row)]]
        scores['nstmix'][num] = scores['nstmix'][num][hypers[np.argmax(row)]]

    # combos = [[400, 0], [400, 1], [400, 2], [400, 3], [400, 4]]
    # for combo in combos:
    #     print('Evaluating for number labeled {} on fold {}'.format(combo[0], combo[1]))
    #     test_file = '/nfs/masi/hansencb/nlst_nst_mixmatch/data_organization/fold{}/test.csv'.format(combo[1])
    #     y_true = []
    #     with open(test_file, 'r') as f:
    #         for line in f.readlines():
    #             y_true.append(int(line.strip().split(',')[-1]))
    #     legend = []
    #     fig, ax = plt.subplots()
    #
    #     exp_scores = {}
    #     patches = []
    #     for method in scores:
    #         exp_scores[method] = scores[method][combo[0]][combo[1]]
    #
    #         # aucvals = aucfun(robjects.FloatVector(exp_scores[method]), robjects.FloatVector(y_true))
    #         # auc = np.array(aucvals[0])
    #         # skauc = sklearn.metrics.roc_auc_score(y_true, exp_scores[method],)
    #         # print('Old AUC {} New AUC {}'.format(auc, skauc))
    #
    #         plotdata = rocplotfun(robjects.FloatVector(exp_scores[method]), robjects.FloatVector(y_true))
    #         fp = np.array(plotdata[0])
    #         tp = np.array(plotdata[1])
    #         regionx = np.array(plotdata[2])
    #         regiony = np.array(plotdata[3])
    #         poly = np.stack((regionx, regiony),1)
    #         patches.append(Polygon(poly))
    #         ax.plot(fp, tp)
    #
    #     p = PatchCollection(patches, cmap='tab20', alpha=0.5)
    #     p.set(array=np.array([0,1,2,3,4]))
    #     ax.add_collection(p)
    #     plt.show()
    #     pass
        # p, z = stat_util.pvalue(y_true, exp_scores['standard'], exp_scores['nstmix'], score_fun=roc_auc_score)
        # print("Standard vs MixMatchNST p-value: {}".format(p/2))
        # break
        # p, z = stat_util.pvalue(y_true, exp_scores['standard'], exp_scores['nst'], score_fun=roc_auc_score)
        # print("Standard vs NST p-value: {}".format(p/2))
        # p, z = stat_util.pvalue(y_true, exp_scores['standard'], exp_scores['nstmix'], score_fun=roc_auc_score)
        # print("Standard vs MixMatchNST p-value: {}".format(p/2))
        # p, z = stat_util.pvalue(y_true, exp_scores['nst'], exp_scores['mix'], score_fun=roc_auc_score)
        # print("NST vs MixMatch p-value: {}".format(p/2))
        # p, z = stat_util.pvalue(y_true, exp_scores['nst'], exp_scores['nstmix'], score_fun=roc_auc_score)
        # print("NST vs MixMatchNST p-value: {}".format(p/2))
        # p, z = stat_util.pvalue(y_true, exp_scores['mix'], exp_scores['nstmix'], score_fun=roc_auc_score)
        # print("MixMatch vs MixMatchNST p-value: {}".format(p/2))
        # print()


    combos = [[200, 0], [200, 1], [200, 2], [200, 3], [200, 4]]

    tmp = []
    for i in range(5):
        tmp.extend(y_true[i])
    y_true = tmp

    for method in scores:
        y = []
        ytop = []
        ybot = []
        for num in nums:
            y_pred = []
            for i in range(5):
                print('Evaluating for number labeled {} on fold {}'.format(num, i))

                y_pred.extend(scores[method][num][i])

            aucvals = aucfun(robjects.FloatVector(y_pred), robjects.FloatVector(y_true))
            auc = np.array(aucvals[0])
            ci = np.array(aucvals[1])
            y.append(auc)
            ybot.append(ci[0])
            ytop.append(ci[1])

        x = np.arange(0, len(nums))
        plt.fill_between(x, ybot, ytop, alpha=0.5)
        plt.plot(x,y)
    plt.legend(list(scores.keys()))
    plt.grid(True)
    plt.xticks(np.arange(0,5), nums)
    plt.xlabel('Num labeled Subjects')
    plt.ylabel('AUC')
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