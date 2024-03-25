import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

fig = plt.figure(figsize=(7.5, 6))
x = [1,2,3,4,5,6,7,8]#点的横坐标
k1 = [0.8741610738255033,0.8758389261744967,0.8640939597315436,0.8708053691275168,0.87751677852349,0.8741610738255033,0.8708053691275168,0.8791946308724832]
k2 = [0.912751677852349,0.9161073825503355,0.9211409395973155,0.9278523489932886,0.9211409395973155,0.9261744966442953,0.9295302013422819,0.9278523489932886]
k3 = [0.9093959731543624,0.9261744966442953,0.9328859060402684,0.933744966442953,0.937080536912751,0.9348859060402684,0.9395302013422819,0.9429530201342282]
print(len(k1))
# k2 = [0.889, 0.918, 0.929, 0.949, 0.971]#线2的纵坐标
# k3 = [0.896, 0.931, 0.946, 0.963, 0.978]
# k4 = [0.862, 0.874, 0.882, 0.909, 0.911]
# k3 = [7.07,10.07,11.4,13.9,14.8,15.1,14.18,15.81,14.04,13.07,13.18,13.49,13.37,13.9]

plt.plot(x,k1,'s-',color = 'r',label="GMF")#s-:方形
plt.plot(x,k2,'o-',color = 'g',label="MLP")#o-:圆形
plt.plot(x,k3,'*-',color = 'b',label="NeuMF")#o-:圆形

plt.xlabel("Epochs", fontdict={'family' : 'Times New Roman', 'size': 23})#横坐标名字
plt.ylabel("HR", fontdict={'family' : 'Times New Roman', 'size': 23})#纵坐标名字
plt.legend(prop={'family' : 'Times New Roman', 'size': 20})#图例
plt.xticks(fontproperties = 'Times New Roman', size=23)
plt.yticks(fontproperties = 'Times New Roman', size=23)
plt.grid()
fig.tight_layout()
plt.ylim(0.81, 0.97)
ax1=plt.gca()
ax1.spines['top'].set_linewidth('1.5')
ax1.spines['bottom'].set_linewidth('1.5')
ax1.spines['left'].set_linewidth('1.5')
ax1.spines['right'].set_linewidth('1.5')
plt.savefig("curve_hr_1.pdf")
plt.show()
