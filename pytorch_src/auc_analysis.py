import matplotlib.pyplot as plt
import os
import numpy as np
import scipy

data = {'standard':{}, 'nst':{}, 'mix':{}, 'nstmix':{}}
fold_track = {'standard':{}, 'nst':{}, 'mix':{}, 'nstmix':{}}
results_dir = '/nfs/masi/hansencb/nlst_nst_mixmatch/results'
nums = [4, 20, 40, 200, 400, 800, 1200, 2000, 2800, 3600]

results_dirs = os.listdir(results_dir)
results_dirs.sort()

for result_dir in results_dirs:
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

    if os.path.isfile(npz_file):
        npz = np.load(npz_file)
        auc = npz['arr_0']
        fpr = npz['arr_1']
        tpr = npz['arr_2']
        scores = npz['arr_3']

        if nst == 0 and lamb == 0:
            if num not in data['standard']:
                data['standard'][num] = np.zeros(5) + np.NaN
            data['standard'][num][fold] = auc
        elif nst != 0 and lamb != 0:
            key = '{}_{}'.format(nst, lamb)
            if num not in data['nstmix']:
                data['nstmix'][num] = {}
            if key not in data['nstmix'][num]:
                data['nstmix'][num][key] = np.zeros(5) + np.NaN
            data['nstmix'][num][key][fold] = auc
        elif nst != 0:
            if num not in data['nst']:
                data['nst'][num] = {}
            if nst not in data['nst'][num]:
                data['nst'][num][nst] = np.zeros(5) + np.NaN
            data['nst'][num][nst][fold] = auc
        else:
            if num not in data['mix']:
                data['mix'][num] = {}
            if lamb not in data['mix'][num]:
                data['mix'][num][lamb] = np.zeros(5) + np.NaN
            data['mix'][num][lamb][fold] = auc

nums = nums[1:-3]

img = []
for num in nums:
    d = data['nst'][num]
    nsts = list(d.keys())
    nsts.sort()

    row = []
    for nst in nsts:
        row.append(np.nanmean(d[nst]))
    img.append(row)

    data['nst'][num] = data['nst'][num][nsts[np.argmax(row)]]


img = []
for num in nums:
    d = data['mix'][num]
    lambs = list(d.keys())
    lambs.sort()

    row = []
    for lamb in lambs:
        row.append(np.nanmean(d[lamb]))
    img.append(row)
    data['mix'][num] = data['mix'][num][lambs[np.argmax(row)]]


img = []
for num in nums:
    d = data['nstmix'][num]
    hypers = list(d.keys())

    row = []
    for hyper in hypers:
        if np.sum(np.isnan(d[hyper]))<2:
            row.append(np.nanmean(d[hyper]))
        else:
            row.append(0)
    img.append(row)
    data['nstmix'][num] = data['nstmix'][num][hypers[np.argmax(row)]]




keys = ['standard', 'mix', 'nst', 'nstmix']
legend = ['Baseline', 'Supervised', 'MixMatch', 'Nullspace Tuning', 'MixMatchNST']


auc_out_file = 'nlst_aucs.csv'

orig_auc = 0.7418
x = np.arange(0, len(nums))
plt.figure()
plt.plot(x, np.zeros(len(nums)) + orig_auc)
# plt.fill_between(x, np.zeros(len(nums)) + orig_auc + 0.0211, np.zeros(len(nums)) + orig_auc -.0211, alpha=0.25)
plt.fill_between(x, np.zeros(len(nums)) + orig_auc + 0, np.zeros(len(nums)) + orig_auc -0, alpha=0.25)

with open(auc_out_file, 'w') as f:
    for i,k in enumerate(keys):
        y = []
        err = []
        f.write('{},fold0,fold1,fold2,fold3,fold4\n'.format(legend[i+1]))
        for n in nums:
            f.write('numsubjects_{},{}\n'.format(n,','.join([str(x) for x in data[k][n]])))
            y.append(np.mean(data[k][n]))
            err.append(np.std(data[k][n])/np.sqrt(len(data[k][n])))




