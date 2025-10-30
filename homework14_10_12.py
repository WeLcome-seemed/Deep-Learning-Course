# 比较深层NN用ReLU 与 浅层NN用sigmoid
import numpy as np
import matplotlib.pyplot as plt

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

w1_s = 2 * np.random.rand(25, hiden_node) - 1
w2_s = 2 * np.random.rand(hiden_node, classnum) - 1

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

def mySigmoid(a):
    return 1/(1 + np.exp(-a))

def Backprop_ReLU(w1, w2, w3, w4, iput, oput):
    total_err = 0
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

        total_err += np.sum(np.abs(e1))
    
    return w1, w2, w3, w4, total_err/5

def Backprop_Softmax(w1_s, w2_s, iput, oput):
    total_err = 0
    for i in range(5):
        v1 = iput[i].reshape(-1) @ w1_s # 1 * hiden_node
        y1 = mySigmoid(v1)
        v2 = y1 @ w2_s # 1 * classnum
        y2 = mySoftmax(v2)

        e1 = oput[i] - y2
        delta1 = e1
        e2 = w2_s @ delta1.reshape(classnum, 1) # hiden_node * 1
        delta2 = y1 * (1 - y1) * e2.reshape(1, hiden_node) # 1 * hiden_node

        w1_s = w1_s + alpha * iput[i].reshape(25, 1) @ delta2
        w2_s = w2_s + alpha * y1.reshape(hiden_node, 1) @ delta1.reshape(1,classnum)

        total_err += np.sum(np.abs(e1))

    return w1_s, w2_s, total_err/5

epochs = 1000
err_ReLU = [0 for _ in range(epochs)]
err_Softmax = [0 for _ in range(epochs)]
for i in range(epochs):
    w1, w2, w3, w4, err_ReLU[i] = Backprop_ReLU(w1, w2, w3, w4, iput, oput)
    w1_s, w2_s, err_Softmax[i] = Backprop_Softmax(w1_s, w2_s, iput, oput)


for i in range(5):
    v1 = iput[i].reshape(-1) @ w1 # 1 * hiden_node
    y1 = myReLU(v1)
    v2 = y1 @ w2
    y2 = myReLU(v2)
    v3 = y2 @ w3
    y3 = myReLU(v3)
    v4 = y3 @ w4
    y4 = mySoftmax(v4)
    print("识别结果为：",np.argmax(y4)+1)

    v1_s = iput[i].reshape(-1) @ w1_s
    y1_s = mySigmoid(v1_s)
    v2_s = y1_s @ w2_s
    y2_s = mySoftmax(v2_s)
    print("识别结果为：",np.argmax(y2_s)+1)

# 画图
epoch = list(range(epochs))
plt.plot(epoch, err_ReLU, label = "ReLU", color = "blue")
plt.plot(epoch, err_Softmax, label = "Softmax", color = "red")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Training Error over Epochs")
plt.legend()
plt.show()
