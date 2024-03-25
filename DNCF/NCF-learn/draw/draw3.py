import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(16, 3.7))
labels = ['4', '8', '16', '32']
zero = [0.87, 0.874, 0.862, 0.869]
first = [0.904, 0.911, 0.909, 0.918]
second = [0.918, 0.908, 0.904, 0.916]
third = [0.925, 0.921, 0.913, 0.929]

plt.subplot(141)
x = np.arange(len(labels))
plt.ylim(0.8, 0.95)
width = 0.18
plt.bar(x - 1.5 * width + 0.01, zero, width,  label='ItemKNN')
plt.bar(x - 0.5 * width + 0.02, first, width,  label='GMF')
plt.bar(x + 0.5 * width + 0.03, second, width, label='MLP')
plt.bar(x + 1.5 * width + 0.04, third, width, label='NeuMF')
plt.ylabel('HR @ MIMIC3-20', fontdict={'family' : 'Times New Roman', 'size': 12})
plt.xlabel('Factors', fontdict={'family' : 'Times New Roman', 'size': 12})
plt.xticks(x, labels=labels, fontproperties = 'Times New Roman', size=12)
plt.yticks(fontproperties = 'Times New Roman', size=12)

ax1=plt.gca()
ax1.spines['top'].set_linewidth('1.3')
ax1.spines['bottom'].set_linewidth('1.3')
ax1.spines['left'].set_linewidth('1.3')
ax1.spines['right'].set_linewidth('1.3')
# plt.legend()

###############################################
labels = ['4', '8', '16', '32']
zero = [0.853, 0.851, 0.838, 0.848]
first= [0.866, 0.871, 0.867, 0.874]
second = [0.873, 0.875, 0.872, 0.882]
third = [0.876, 0.881, 0.873, 0.889]

plt.subplot(142)
x = np.arange(len(labels))  # x轴刻度标签位置
plt.ylim(0.8, 0.9)
plt.bar(x - 1.5 * width + 0.01, zero, width,  label='ItemKNN')
plt.bar(x - 0.5 * width + 0.02, first, width,  label='GMF')
plt.bar(x + 0.5 * width + 0.03, second, width, label='MLP')
plt.bar(x + 1.5 * width + 0.04, third, width, label='NeuMF')
plt.ylabel('NDCG @ MIMIC3-20', fontdict={'family' : 'Times New Roman', 'size': 12})
plt.xlabel('Factors', fontdict={'family' : 'Times New Roman', 'size': 12})
plt.xticks(x, labels=labels, fontproperties = 'Times New Roman', size=12)
plt.yticks(fontproperties = 'Times New Roman', size=12)
ax1=plt.gca()
ax1.spines['top'].set_linewidth('1.3')
ax1.spines['bottom'].set_linewidth('1.3')
ax1.spines['left'].set_linewidth('1.3')
ax1.spines['right'].set_linewidth('1.3')
# plt.legend()


###################################################
labels = ['4', '8', '16', '32']
zero = [0.844, 0.846, 0.849, 0.852]
first = [0.8792, 0.8842, 0.8758, 0.8859]
second = [0.9128, 0.9094, 0.9228, 0.9178]
third = [0.9144, 0.9279, 0.9295, 0.9329]
plt.subplot(143)
x = np.arange(len(labels))  # x轴刻度标签位置
plt.ylim(0.8, 0.95)
plt.bar(x - 1.5 * width + 0.01, zero, width,  label='ItemKNN')
plt.bar(x - 0.5 * width + 0.02, first, width,  label='GMF')
plt.bar(x + 0.5 * width + 0.03, second, width, label='MLP')
plt.bar(x + 1.5 * width + 0.04, third, width, label='NeuMF')
plt.ylabel('HR @ MIMIC3-30', fontdict={'family' : 'Times New Roman', 'size': 12})
plt.xlabel('Factors', fontdict={'family' : 'Times New Roman', 'size': 12})
plt.xticks(x, labels=labels, fontproperties = 'Times New Roman', size=12)
plt.yticks(fontproperties = 'Times New Roman', size=12)
# plt.legend()
ax1=plt.gca()
ax1.spines['top'].set_linewidth('1.3')
ax1.spines['bottom'].set_linewidth('1.3')
ax1.spines['left'].set_linewidth('1.3')
ax1.spines['right'].set_linewidth('1.3')

###################################################
labels = ['4', '8', '16', '32']
zero = [0.817, 0.824, 0.829, 0.833]
first = [0.8411, 0.8426, 0.8353, 0.846]
second = [0.8774, 0.8755, 0.8904, 0.8804]
third = [0.8733, 0.8933, 0.9009, 0.8981]

plt.subplot(144)
x = np.arange(len(labels))  # x轴刻度标签位置
plt.ylim(0.8, 0.92)
plt.bar(x - 1.5 * width + 0.01, zero, width,  label='ItemKNN')
plt.bar(x - 0.5 * width + 0.02, first, width,  label='GMF')
plt.bar(x + 0.5 * width + 0.03, second, width, label='MLP')
plt.bar(x + 1.5 * width + 0.04, third, width, label='NeuMF')
plt.ylabel('NDCG @ MIMIC3-30', fontdict={'family' : 'Times New Roman', 'size': 12})
plt.xlabel('Factors', fontdict={'family' : 'Times New Roman', 'size': 12})
plt.xticks(x, labels=labels, fontproperties = 'Times New Roman', size=12)
plt.yticks(fontproperties = 'Times New Roman', size=12)
# plt.legend()
ax1=plt.gca()
ax1.spines['top'].set_linewidth('1.3')
ax1.spines['bottom'].set_linewidth('1.3')
ax1.spines['left'].set_linewidth('1.3')
ax1.spines['right'].set_linewidth('1.3')


plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0,
           prop={'family' : 'Times New Roman', 'size': 10})
fig.tight_layout()
plt.savefig("ablation_matrix.pdf", bbox_inches='tight')
plt.show()
