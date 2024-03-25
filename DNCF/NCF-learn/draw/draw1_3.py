import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

fig = plt.figure(figsize=(7.5, 6))
x = [1,2,3,4,5,6,7,8,9,10,11,12]#点的横坐标
k1 = [0.043446135479886246,0.016557788831662933,0.0140413361972295,0.012514889966343392,0.011276374255986751,0.010425854795082732,0.009677503992068489,0.00911238888980406,0.008601911794570809,0.008265635786882615,0.00789567894971007,0.007601094457609976]#线1的纵坐标
print(len(k1))
# k2 = [0.889, 0.918, 0.929, 0.949, 0.971]#线2的纵坐标
# k3 = [0.896, 0.931, 0.946, 0.963, 0.978]
# k4 = [0.862, 0.874, 0.882, 0.909, 0.911]
# k3 = [7.07,10.07,11.4,13.9,14.8,15.1,14.18,15.81,14.04,13.07,13.18,13.49,13.37,13.9]

plt.plot(x,k1,'s-',color = 'r',label="GMF")#s-:方形
# plt.plot(x,k2,'o-',color = 'g',label="MLP")#o-:圆形
# plt.plot(x,k3,'+-',color = 'b',label="NeuMF")#o-:圆形

plt.xlabel("Top-K", fontdict={'family' : 'Times New Roman', 'size': 23})#横坐标名字
plt.ylabel("HR@MIMIC3-20", fontdict={'family' : 'Times New Roman', 'size': 23})#纵坐标名字
plt.legend(prop={'family' : 'Times New Roman', 'size': 20})#图例
# plt.title("Air-quality", fontname="Times New Roman", fontsize=15)
plt.xticks(fontproperties = 'Times New Roman', size=23)
plt.yticks(fontproperties = 'Times New Roman', size=23)
# plt.grid()
fig.tight_layout()
# plt.ylim(0.83, 0.99)
ax1=plt.gca()
ax1.spines['top'].set_linewidth('1.5')
ax1.spines['bottom'].set_linewidth('1.5')
ax1.spines['left'].set_linewidth('1.5')
ax1.spines['right'].set_linewidth('1.5')
plt.savefig("curve_HR.pdf")
plt.show()
