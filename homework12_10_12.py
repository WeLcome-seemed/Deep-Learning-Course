#补全激活函数为ReLU的数字识别代码，观察是否有效(3隐藏层)
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
# w1 = 2 * np.random.rand(25, hiden_node) - 1
# w2 = 2 * np.random.rand(hiden_node, hiden_node) - 1
# w3 = 2 * np.random.rand(hiden_node, hiden_node) - 1
# w4 = 2 * np.random.rand(hiden_node, classnum) - 1

# He初始化： W ~ N(0, √(2/n_in)) 或 W ~ U(-√(6/n_in), √(6/n_in))
w1 = np.random.randn(25, hiden_node) * np.sqrt(2.0 / 25)
w2 = np.random.randn(hiden_node, hiden_node) * np.sqrt(2.0 / hiden_node)
w3 = np.random.randn(hiden_node, hiden_node) * np.sqrt(2.0 / hiden_node)
w4 = np.random.randn(hiden_node, classnum) * np.sqrt(2.0 / hiden_node)
alpha = 0.1

def myReLU(a):
    return (a > 0) * a

def mySoftmax(a):
    a_shifted = a - np.max(a) # 防止溢出，增加数值稳定性
    ex = np.exp(a_shifted)
    y = ex / np.sum(ex)
    return y

def Backprop(w1, w2, w3, w4, iput, oput):
    for i in range(5):
        v1 = iput[i].reshape(-1) @ w1 # 1 * hiden_node
        y1 = myReLU(v1)
        v2 = y1 @ w2 # 1 * hiden_node
        y2 = myReLU(v2)
        v3 = y2 @ w3 # 1 * hiden_node
        y3 = myReLU(v3)
        v4 = y3 @ w4 # 1 * classnum
        y4 = mySoftmax(v4)

        e1 = oput[i] - y4
        delta1 = e1 # 1 * classnum
        e2 = w4 @ delta1.reshape(classnum, 1) # hiden_node * 1
        delta2 = (v3 > 0) * e2.reshape(1, hiden_node) # 1 * hiden_node
        e3 = w3 @ delta2.reshape(hiden_node, 1) # hiden_node * 1
        delta3 = (v2 > 0) * e3.reshape(1, hiden_node) # 1 * hiden_node
        e4 = w2 @ delta3.reshape(hiden_node, 1) # hiden_node * 1
        delta4 = (v1 > 0) * e4.reshape(1, hiden_node) # 1 * hiden _ndoe

        w1 = w1 + alpha * iput[i].reshape(25, 1) @ delta4 # (25, 1) * (1, hiden_node)
        w2 = w2 + alpha * y1.reshape(hiden_node, 1) @ delta3 # (hiden_node, 1) * (1, hiden_node)
        w3 = w3 + alpha * y2.reshape(hiden_node, 1) @ delta2 # (hiden_node, 1) * (1, hiden_ndoe)
        w4 = w4 + alpha * y3.reshape(hiden_node, 1) @ delta1.reshape(1,classnum) # (hiden_node, 1) * (1, classnum)
    
    return w1, w2, w3, w4

for _ in range(5000):
    w1, w2, w3, w4 = Backprop(w1, w2, w3, w4, iput, oput)


for i in range(5):
    v1 = iput[i].reshape(-1) @ w1 # 1 * hiden_node
    y1 = myReLU(v1)
    v2 = y1 @ w2
    y2 = myReLU(v2)
    v3 = y2 @ w3
    y3 = myReLU(v3)
    v4 = y3 @ w4
    y4 = mySoftmax(v4)
    print(y4)