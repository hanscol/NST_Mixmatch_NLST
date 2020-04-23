import matplotlib.pyplot as plt
import os
import numpy as np
import scipy

data = {'standard':{}, 'nst':{}, 'mix':{}, 'nstmix':{}}
results_dir = '/nfs/masi/hansencb/nlst_nst_mixmatch/results'
# nums = [1599, 1998, 2397, 2797, 3597]
nums = [1599, 2397, 3597]
# nums = [40, 200, 400]
for result_dir in os.listdir(results_dir):
    parts = result_dir.split('_')
    # key = '_'.join(parts[1:])

    fold = int(parts[0].lstrip('fold'))
    num = int(parts[1].split('d')[-1])
    nst = float(parts[2].split('a')[-1])
    lamb = float(parts[3].split('a')[-1])

    if fold <= 4:

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
                    data['standard'][num] = []
                data['standard'][num].append(auc)
            elif nst != 0 and lamb != 0:
                key = '{}_{}'.format(nst, lamb)
                if num not in data['nstmix']:
                    data['nstmix'][num] = {}
                if key not in data['nstmix'][num]:
                    data['nstmix'][num][key] = []
                data['nstmix'][num][key].append(auc)
            elif nst != 0:
                if num not in data['nst']:
                    data['nst'][num] = {}
                if nst not in data['nst'][num]:
                    data['nst'][num][nst] = []
                data['nst'][num][nst].append(auc)
            else:
                if num not in data['mix']:
                    data['mix'][num] = {}
                if lamb not in data['mix'][num]:
                    data['mix'][num][lamb] = []
                data['mix'][num][lamb].append(auc)

# nums = nums[1:-3]

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
    print('Number labeled {} Best nstlambda {}'.format(num, nsts[np.argmax(row)]))

img = np.array(img)

# plt.imshow(img)
# plt.yticks(np.arange(0, len(nums)), nums)
# plt.xticks(np.arange(0, len(nsts)), nsts)
# plt.ylabel('Number of Labeled Subjects')
# plt.xlabel('Nullspace Tuning Lambda')
# plt.colorbar()
# plt.show()

# img = []
# for num in nums:
#     d = data['mix'][num]
#     lambs = list(d.keys())
#     lambs.sort()
#
#     row = []
#     for lamb in lambs:
#         row.append(np.mean(d[lamb]))
#     img.append(row)
#     data['mix'][num] = data['mix'][num][lambs[np.argmax(row)]]
#     print('Number labeled {} Best lambda {}'.format(num, lambs[np.argmax(row)]))
#
# img = np.array(img)

# plt.imshow(img)
# plt.yticks(np.arange(0, len(nums)), nums)
# plt.xticks(np.arange(0, len(lambs)), lambs)
# plt.ylabel('Number of Labeled Subjects')
# plt.xlabel('MixMatch Tuning Lambda')
# plt.colorbar()
# plt.show()


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
    print('Number labeled {} Best hyper {}'.format(num, hypers[np.argmax(row)]))



# keys = ['standard', 'mix', 'nst', 'nstmix']
# legend = ['Baseline', 'Supervised', 'MixMatch', 'Nullspace Tuning', 'MixMatchNST']

keys = ['standard', 'nst', 'nstmix']
legend = ['Baseline', 'Supervised', 'Nullspace Tuning', 'MixMatchNST']

auc_out_file = 'nlst_aucs.csv'

orig_auc = 0.7757
x = np.arange(0, len(nums))
plt.figure()
plt.plot(x, np.zeros(len(nums)) + orig_auc)
# plt.fill_between(x, np.zeros(len(nums)) + orig_auc + 0.0211, np.zeros(len(nums)) + orig_auc -.0211, alpha=0.25)
plt.fill_between(x, np.zeros(len(nums)) + orig_auc + 0, np.zeros(len(nums)) + orig_auc -0, alpha=0.25)

with open(auc_out_file, 'w') as f:
    for i,k in enumerate(keys):
        y = []
        err = []
        f.write('{}\n'.format(legend[i+1]))
        for n in nums:
            f.write('{}\n'.format(','.join([str(x) for x in data[k][n]])))
            y.append(np.mean(data[k][n]))
            err.append(np.std(data[k][n])/np.sqrt(len(data[k][n])))

        y, err = np.array(y), np.array(err)
        plt.plot(x, y)
        plt.fill_between(x, y+err, y-err, alpha=0.25)

plt.legend(legend, loc='lower right')
plt.grid(True)
plt.xlabel('Number of Labeled Subjects')
plt.ylabel('AUC')
plt.xticks(x, nums)
plt.show()



