import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(7.5, 6))
x = [2,3,4,5,6]#点的横坐标
k1 = [0.863, 0.873, 0.877, 0.887, 0.891]#线1的纵坐标
k2 = [0.869, 0.882, 0.886, 0.894, 0.898]#线2的纵坐标
k3 = [0.881, 0.890, 0.894, 0.899, 0.909]
k4 = [0.853, 0.858, 0.862, 0.868, 0.881]
# k3 = [7.07,10.07,11.4,13.9,14.8,1 5.1,14.18,15.81,14.04,13.07,13.18,13.49,13.37,13.9]

plt.plot(x,k1,'s-',color = 'r',label="GMF")#s-:方形
plt.plot(x,k2,'o-',color = 'g',label="MLP")#o-:圆形
plt.plot(x,k3,'*-',color = 'b',label="NeuMF")#o-:圆形
plt.plot(x,k4,'+-', color = 'c', label="ItemKNN")#o-:圆形
plt.xlabel("Top-K", fontdict={'family' : 'Times New Roman', 'size': 23})#横坐标名字
plt.ylabel("NDCG@MIMIC3-20", fontdict={'family' : 'Times New Roman', 'size': 23})#纵坐标名字
plt.legend(prop={'family' : 'Times New Roman', 'size': 20})#图例
# plt.title("Air-quality", fontname="Times New Roman", fontsize=15)
plt.xticks(fontproperties = 'Times New Roman', size=23)
plt.yticks(fontproperties = 'Times New Roman', size=23)
plt.grid()
fig.tight_layout()
plt.ylim(0.82, 0.92)
ax1=plt.gca()
ax1.spines['top'].set_linewidth('1.5')
ax1.spines['bottom'].set_linewidth('1.5')
ax1.spines['left'].set_linewidth('1.5')
ax1.spines['right'].set_linewidth('1.5')
plt.savefig("curve_ndcg.pdf")
plt.show()
