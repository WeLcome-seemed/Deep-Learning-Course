# 分别用交叉熵和误差平方和代价函数训练同一神经网络求解XOR问题，比较误差-轮曲线
import numpy as np
import matplotlib.pyplot as plt

iput = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
oput = np.array([0, 1, 1, 0])
hiden_node = 4
w1 = 2 * np.random.rand(hiden_node, 3) - 1
w2 = 2 * np.random.rand(1, hiden_node) - 1
alpha = 0.9
beta = 0.9
mmt1 = np.zeros(hiden_node*3).reshape(hiden_node,3)
mmt2 = np.zeros(hiden_node).reshape(1,hiden_node)

def mySigmoid(a):
    return 1/(1 + np.exp(-a))

#误差平方和代价函数
def backpropXOR_SSE(w1, w2, iput, oput, mmt1, mmt2):
    for i in range(4):
        v1 = iput[i] @ w1.T
        y1 = mySigmoid(v1)  # 1*4
        v2 = y1 @ w2.reshape(hiden_node, 1)
        y2 = mySigmoid(v2)
        e1 = oput[i] - y2
        delta1 = y2 * (1 - y2) * e1
        e2 = w2 * delta1    # 1*4
        delta2 = y1 * (1 - y1) * e2     # 1*4
        mmt2 = alpha * delta1 * y1 + beta * mmt2
        w2 = w2 + mmt2
        mmt1 = alpha * (iput[i].reshape(3, 1) @ delta2).T + beta * mmt1
        w1 = w1 + mmt1
    return w1, w2, np.abs(e1), mmt1, mmt2

#交叉熵代价函数
def backpropXOR_CE(w1, w2, iput, oput, mmt1, mmt2):
    for i in range(4):
        v1 = iput[i] @ w1.T
        y1 = mySigmoid(v1)
        v2 = y1 @ w2.reshape(hiden_node, 1)
        y2 = mySigmoid(v2)
        e1 = oput[i] - y2
        delta1 = e1
        e2 = w2 * e1 
        delta2 = y1 * (1 - y1) * e2
        mmt2 = alpha * delta1 * y1 + beta * mmt2
        w2 = w2 + mmt2
        mmt1 = alpha * (iput[i].reshape(3, 1) @ delta2).T + beta * mmt1
        w1 = w1 + mmt1
    return w1, w2, np.abs(e1), mmt1, mmt2

w1_SSE, w1_CE = w1.copy(), w1.copy(); w2_SSE, w2_CE = w2.copy(), w2.copy()
err_SSE = [0 for i in range(8000)]
err_CE = [0 for i in range(8000)]
for i in range(8000):
    w1_SSE, w2_SSE, err_SSE[i], mmt1, mmt2 = backpropXOR_SSE(w1_SSE, w2_SSE, iput, oput, mmt1, mmt2)
mmt1 = np.zeros(hiden_node*3).reshape(hiden_node,3)
mmt2 = np.zeros(hiden_node).reshape(1,hiden_node)    
for i in range(8000):
    w1_CE, w2_CE, err_CE[i], mmt1, mmt2 = backpropXOR_CE(w1_CE, w2_CE, iput, oput, mmt1, mmt2)

v1_SSE = iput @ w1_SSE.T    # 4*4
v1_CE = iput @ w1_CE.T
y1_SSE = mySigmoid(v1_SSE)
y1_CE = mySigmoid(v1_CE)
v2_SSE = y1_SSE @ w2_SSE.reshape(hiden_node, 1) # 4*1
v2_CE = y1_CE @ w2_CE.reshape(hiden_node, 1)
y2_SSE = mySigmoid(v2_SSE)
y2_CE = mySigmoid(v2_CE)

#画图
epochs = list(range(1,8001))
plt.plot(epochs, err_SSE, label = "SSE", color = "blue")
plt.plot(epochs, err_CE, label = "CE", color = "red")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Training Error over Epochs")
plt.legend()
plt.show()

