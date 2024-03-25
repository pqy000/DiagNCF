import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

fig = plt.figure(figsize=(7.5, 6))
x = [1,2,3,4,5,6,7,8]#点的横坐标
k1 = [0.12193142463192697,0.03173786770305221,0.025728192913977172,0.020075923573861332,0.017280102396710507,0.015804715164367693,0.014658485565299247,0.013520300332942351]
k2 = [0.04186917982015635,0.01678858716835343,0.014881519746626793,0.013857123999242887,0.013113905746330864,0.012379953314735603,0.011817271105494898,0.011267211938263645]
k3 = [0.0403681173814699,0.01452673085704799,0.012273755505288204,0.011039227315983889,0.010126637982456147,0.009386598081570736,0.008829579683256853,0.00826392148911591]
print(len(k1))
# k2 = [0.889, 0.918, 0.929, 0.949, 0.971]#线2的纵坐标
# k3 = [0.896, 0.931, 0.946, 0.963, 0.978]
# k4 = [0.862, 0.874, 0.882, 0.909, 0.911]
# k3 = [7.07,10.07,11.4,13.9,14.8,15.1,14.18,15.81,14.04,13.07,13.18,13.49,13.37,13.9]

plt.plot(x,k1,'s-',color = 'r',label="GMF")#s-:方形
plt.plot(x,k2,'o-',color = 'g',label="MLP")#o-:圆形
plt.plot(x,k3,'*-',color = 'b',label="NeuMF")#o-:圆形

plt.xlabel("Epochs", fontdict={'family' : 'Times New Roman', 'size': 23})#横坐标名字
plt.ylabel("Training loss", fontdict={'family' : 'Times New Roman', 'size': 23})#纵坐标名字
plt.legend(prop={'family' : 'Times New Roman', 'size': 20})#图例
plt.xticks(fontproperties = 'Times New Roman', size=23)
plt.yticks(fontproperties = 'Times New Roman', size=23)
plt.grid()
fig.tight_layout()
# plt.ylim(0.83, 0.99)
ax1=plt.gca()
ax1.spines['top'].set_linewidth('1.5')
ax1.spines['bottom'].set_linewidth('1.5')
ax1.spines['left'].set_linewidth('1.5')
ax1.spines['right'].set_linewidth('1.5')
plt.savefig("curve_training.pdf")
plt.show()
