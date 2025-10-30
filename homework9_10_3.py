#设计网络结构实现对低分辨率数字图像的分类
import numpy as np

iput = np.random.rand(5, 5, 5)
iput[0] =  [[0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0]]
iput[1] =  [[1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1]]
iput[2] =  [[1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0]]
iput[3] =  [[0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0]]
iput[4] =  [[1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0]]
oput =     [[1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]]
hiden_node = 25 * 2
classnum = 5
w1 = 2 * np.random.rand(25, hiden_node) - 1
w2 = 2 * np.random.rand(hiden_node, classnum) - 1
alpha = 0.9

def mySigmoid(a):
    return 1/(1 + np.exp(-a))

def mySoftmax(a):
    ex = np.exp(a)
    y = ex / np.sum(ex)
    return y

def Backprop(w1, w2, iput, oput):
    for i in range(5):
        v1 = iput[i].reshape(-1) @ w1 # 1 * hiden_node
        y1 = mySigmoid(v1)
        v2 = y1 @ w2 # 1 * classnum
        y2 = mySoftmax(v2)
        e1 = oput[i] - y2
        delta1 = e1
        e2 = w2 @ delta1.reshape(classnum, 1) # hiden_node * 1
        delta2 = y1 * (1 - y1) * e2.reshape(1, hiden_node) # 1 * hiden_node
        w1 = w1 + alpha * iput[i].reshape(25, 1) @ delta2
        w2 = w2 + alpha * y1.reshape(hiden_node, 1) @ delta1.reshape(1,classnum)
    return w1, w2

for _ in range(5000):
    w1, w2 = Backprop(w1, w2, iput, oput)


for i in range(5):
    v1 = iput[i].reshape(-1) @ w1 # 1 * hiden_node
    y1 = mySigmoid(v1)
    v2 = v1 @ w2 # 1 * classnum
    y2 = mySoftmax(v2)
    print(y2)