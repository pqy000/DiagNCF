import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(12.3, 3.7))
labels = ["SegNet", "FCN8", "Bayes-Seg", "Eigen", "MCDropout", "Ours"]
zero = [66.1]
first = [61.8]
second = [68]
third = [65.6]
four = [70.6]
five = [72.3]

x = np.arange(len(labels))
width = 0.18
plt.bar(x, zero, width,  label='SegNet')
# plt.bar(x - 2.5 * width + 0.01, zero, width,  label='SegNet')
# plt.bar(x - 1.5 * width + 0.02, first, width,  label='FCN-8')
# plt.bar(x - 0.5 * width + 0.03, second, width,  label='Bayes-Seg')
# plt.bar(x + 0.5 * width + 0.04, third, width, label='Eigen')
# plt.bar(x + 1.5 * width + 0.05, four, width, label='MC-Dropout')
# plt.bar(x + 2.5 * width + 0.06, five, width, label='Ours')
plt.ylabel('Negative-Log-likelihood', fontdict={'family' : 'Times New Roman', 'size': 12})
plt.xticks(x, labels=labels[0], fontproperties = 'Times New Roman', size=12)
plt.yticks(fontproperties = 'Times New Roman', size=12)
plt.ylim(0, -19)
ax1=plt.gca()
ax1.spines['top'].set_linewidth('1.3')
ax1.spines['bottom'].set_linewidth('1.3')
ax1.spines['left'].set_linewidth('1.3')
ax1.spines['right'].set_linewidth('1.3')
plt.show()
# plt.legend()

###############################################
labels = ['Air-quality', 'Bicycle', 'PM2.5', 'Exchange']
zero = [0.9034, 0.9147, 0.7013, 0.9161]
first = [0.906, 0.9123, 0.7011, 0.9393]
second = [0.9094, 0.9116, 0.6992, 0.9323]
third = [0.9161, 0.9141, 0.7072, 0.9345]
plt.subplot(132)
x = np.arange(len(labels))  # x轴刻度标签位置
plt.ylim(0.67, 1.00)
plt.bar(x - 1.5 * width + 0.01, zero, width,  label='$\Sigma_{low_{old}}$')
plt.bar(x - 0.5 * width + 0.02, first, width,  label='$\Sigma_{diag}$')
plt.bar(x + 0.5 * width + 0.03, second, width, label='$\Sigma_{full}$')
plt.bar(x + 1.5 * width + 0.04, third, width, label='$\Sigma_{all}$')
plt.ylabel('CORR', fontdict={'family' : 'Times New Roman', 'size': 12})
plt.xticks(x, labels=labels, fontproperties = 'Times New Roman', size=12)
plt.yticks(fontproperties = 'Times New Roman', size=12)
ax1=plt.gca()
ax1.spines['top'].set_linewidth('1.3')
ax1.spines['bottom'].set_linewidth('1.3')
ax1.spines['left'].set_linewidth('1.3')
ax1.spines['right'].set_linewidth('1.3')
# plt.legend()


###################################################
labels = ['Air-quality', 'Bicycle', 'PM2.5', 'Exchange']
zero = [0.05876, 0.07827, 0.1021, 0.0616]
first = [0.059, 0.07956, 0.1059, 0.052]
second = [0.0579, 0.07664, 0.1074, 0.058]
third = [0.0568, 0.0762, 0.1046, 0.04276]
plt.subplot(133)
x = np.arange(len(labels))  # x轴刻度标签位置
plt.ylim(0.025, 0.12)
plt.bar(x - 1.5 * width + 0.01, zero, width,  label='$\Sigma_{low_{old}}$')
plt.bar(x - 0.5 * width + 0.02, first, width,  label='$\Sigma_{diag}$')
plt.bar(x + 0.5 * width + 0.03, second, width, label='$\Sigma_{low}$')
plt.bar(x + 1.5 * width + 0.04, third, width, label='$\Sigma_{all}$')
plt.ylabel('RMSE', fontdict={'family' : 'Times New Roman', 'size': 12})
# plt.title('CORR')
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
