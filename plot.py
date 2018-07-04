import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glob import glob
import json
import os
import sys
import crash_on_ipy
# sys.path.append(os.path.abspath(os.getcwd() + "./../"))

# files = glob("./repeat-copy/*-{}.json".format(batch_num))
# name='demo-epoch-239'
#
# fname = "./"+ name +'.json'
fname = 'agru_lm_coef_preproc-epoch-74.json'
# files.append("./copy-task-test-10-batch-{}.json".format(batch_num))

# Read the metrics from the .json files
history = json.loads(open(fname, "rt").read())

# interval=len(history['cost'])/len(history['valid_accurs'])
# valid_accurs=[]
# for accur in history[0]['valid_accurs']:
#     valid_accurs.extend(list(np.repeat(accur,interval)))
# valid_accurs = np.array([valid_accurs])
# training = np.array([(x['cost'], x['loss'], x['seq_lengths']) for x in history])
# training_a = np.vstack((training,valid_accurs))
# training = []
# for (x, accur) in zip(history):
#     training.append((x['cost'], x['loss'], x['seq_lengths'], accur))
# training = np.array(training)
training = np.array(history['loss'])
f1 = np.array(history['f1s'])
print(np.max(f1))
f1 = f1.reshape(-1)

print(np.where(f1 == np.max(f1)), np.max(f1))
print(np.where(f1 >= 0.60))

print("Training history (seed x metric x sequence) =", training.shape)

# Average every dv values across each (seed, metric)
dv = 1
l = int(len(training)/dv)
training_mean = training[:l*dv].reshape(-1, dv).mean(axis=1)
f1_mean = f1[:l*dv].reshape(-1, dv).mean(axis=1)
training_std = training[:l*dv].reshape(-1, dv).std(axis=1)
# print(training.shape)

# Average the seeds
fig = plt.figure(figsize=(14, 5))
batch_num=64
# X axis is normalized to thousands
x = np.arange(l)

# Plot the cost
# plt.plot(x, training_mean[0], 'o-', linewidth=2, label='Cost')
plt.errorbar(x, training_mean, fmt='o-', elinewidth=2, linewidth=2, label='loss_mean')
plt.errorbar(x, f1_mean, fmt='x-', elinewidth=1, linewidth=1, label='f1')
# plt.errorbar(x, training_mean+training_std, fmt='x-', elinewidth=2, linewidth=2, label='loss+std')
# plt.errorbar(x, training_mean-training_std, fmt='x-', elinewidth=2, linewidth=2, label='loss-std')
# print(training_mean)
# plt.errorbar(x, training_mean[3], yerr=training_std[3], fmt='r-', elinewidth=2, linewidth=2, label='Accur')
plt.grid()
plt.xticks(np.arange(0, 100, 5))
plt.yticks(np.arange(0, 2, 0.1))
plt.ylabel('Loss with standard deviation')
plt.xlabel('Epoch')
plt.title('Loss vs. valid F1', fontsize=16)

plt.show()

# ax = plt.axes([.57, .55, .25, .25], facecolor=(0.97, 0.97, 0.97))
# plt.title("BCELoss/Accuracy")
# plt.plot(x, training_mean[1], 'r-', label='BCE Loss')
# plt.plot(x, training_mean[3], 'x-', label='Accur')
# # plt.yticks(np.arange(0, training_mean[1][0]+0.2, 0.05))
# # plt.yticks(np.arange(0, training_mean[1][0]+0.2, 0.2))
# plt.yticks(np.arange(0, 1, 0.2))
# plt.grid()
#
# plt.show()
#
# loss = history[0]['loss']
# cost = history[0]['cost']
# seq_lengths = history[0]['seq_lengths']
#
# unique_sls = set(seq_lengths)
# all_metric = list(zip(range(1, batch_num + 1), seq_lengths, loss, cost))
#
# fig = plt.figure(figsize=(12, 5))
# plt.ylabel('Cost per sequence (bits)')
# plt.xlabel('Iteration (thousands)')
# plt.title('Training Convergence (Per Sequence Length)', fontsize=16)
#
# for sl in unique_sls:
#     sl_metrics = [i for i in all_metric if i[1] == sl]
#
#     x = [i[0] for i in sl_metrics]
#     y = [i[3] for i in sl_metrics]
#
#     num_pts = len(x) // 50
#     total_pts = num_pts * 50
#
#     x_mean = [i.mean() / 1000 for i in np.split(np.array(x)[:total_pts], num_pts)]
#     y_mean = [i.mean() for i in np.split(np.array(y)[:total_pts], num_pts)]
#
#     plt.plot(x_mean, y_mean, label='Seq-{}'.format(sl))
#
# plt.yticks(np.arange(0, 80, 5))
# plt.legend(loc=0)
# plt.show()

