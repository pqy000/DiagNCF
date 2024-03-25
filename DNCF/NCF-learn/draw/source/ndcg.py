import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

fig = plt.figure(figsize=(7.5, 6))
x = [1,2,3,4,5,6,7,8]#点的横坐标
k1 = [0.8275393810882088,0.8250623995685542,0.8191501193432424,0.8248029217701359,0.8307152019954478,0.8271000194990429,0.8267004549137122,0.8342507904841819]
k2 = [0.8578203173777587,0.8636530035954,0.8666489407118909,0.8736198279062825,0.8702243262044607,0.8748583186661097,0.8771554163950166,0.8735402338986118]
k3 = [0.8571214779901744,0.874299566065438,0.8796721949185065,0.8860968094259372,0.885151353511508,0.884406477163233,0.8835111210930033,0.8899987868108847]
print(len(k1))

plt.plot(x,k1,'s-',color = 'r',label="GMF")#s-:方形
plt.plot(x,k2,'o-',color = 'g',label="MLP")#o-:圆形
plt.plot(x,k3,'*-',color = 'b',label="NeuMF")#o-:圆形

plt.xlabel("Epochs", fontdict={'family' : 'Times New Roman', 'size': 23})#横坐标名字
plt.ylabel("NDCG", fontdict={'family' : 'Times New Roman', 'size': 23})#纵坐标名字
plt.legend(prop={'family' : 'Times New Roman', 'size': 20})#图例
plt.xticks(fontproperties = 'Times New Roman', size=23)
plt.yticks(fontproperties = 'Times New Roman', size=23)
plt.grid()
fig.tight_layout()
plt.ylim(0.75, 0.92)
ax1=plt.gca()
ax1.spines['top'].set_linewidth('1.5')
ax1.spines['bottom'].set_linewidth('1.5')
ax1.spines['left'].set_linewidth('1.5')
ax1.spines['right'].set_linewidth('1.5')
plt.savefig("curve_ndcg.pdf")
plt.show()
