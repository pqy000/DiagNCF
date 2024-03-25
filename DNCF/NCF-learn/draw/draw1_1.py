import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

fig = plt.figure(figsize=(7.5, 6))
x = [2,3,4,5,6]#点的横坐标
k1 = [0.871,0.914,0.936,0.956,0.966]#线1的纵坐标
k2 = [0.889, 0.918, 0.929, 0.949, 0.971]#线2的纵坐标
k3 = [0.896, 0.931, 0.946, 0.963, 0.978]
k4 = [0.862, 0.874, 0.882, 0.909, 0.911]
# k3 = [7.07,10.07,11.4,13.9,14.8,15.1,14.18,15.81,14.04,13.07,13.18,13.49,13.37,13.9]
# for i in range(len(k1)):
#     k1[i] = k1[i] * -1
#     k2[i] = k2[i] * -1
#     k3[i] = k3[i] * -1
#     k4[i] = k4[i] * -1
# plt.plot(x,k3,'p-',color = 'b',label="MC-Evidencial")#o-:圆形
plt.plot(x,k1,'s-',color = 'r',label="GMF")#s-:方形
plt.plot(x,k2,'o-',color = 'g',label="MLP")#o-:圆形
plt.plot(x,k3,'*-',color = 'b',label="NeuMF")#o-:圆形
plt.plot(x,k4,'+-',color = 'c',label="ItemKNN")#o-:圆形
plt.xlabel("Top-K", fontdict={'family' : 'Times New Roman', 'size': 23})#横坐标名字
plt.ylabel("HR@MIMIC3-20", fontdict={'family' : 'Times New Roman', 'size': 23})#纵坐标名字
plt.legend(prop={'family' : 'Times New Roman', 'size': 20})#图例
# plt.title("Air-quality", fontname="Times New Roman", fontsize=15)
plt.xticks(fontproperties = 'Times New Roman', size=23)
plt.yticks(fontproperties = 'Times New Roman', size=23)
plt.grid()
fig.tight_layout()
plt.ylim(0.83, 0.99)
ax1=plt.gca()
ax1.spines['top'].set_linewidth('1.5')
ax1.spines['bottom'].set_linewidth('1.5')
ax1.spines['left'].set_linewidth('1.5')
ax1.spines['right'].set_linewidth('1.5')
plt.savefig("curve_HR.pdf")
plt.show()
