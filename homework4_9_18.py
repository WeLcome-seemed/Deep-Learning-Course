#训练浅层NN解决XOR问题(1隐藏层)
import numpy as np

iput = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
oput = np.array([0, 1, 1, 0])
hiden_node = 4
w1 = 2 * np.random.rand(hiden_node, 3) - 1
w2 = 2 * np.random.rand(1, hiden_node) - 1
alpha = 0.9

def mySigmoid(a):
    return 1/(1 + np.exp(-a))

def backpropXOR(w1, w2, iput, oput):
    for i in range(4):
        v1 = iput[i] @ w1.T
        y1 = mySigmoid(v1)  # 1*4
        v2 = y1 @ w2.reshape(hiden_node, 1)
        y2 = mySigmoid(v2)
        e1 = oput[i] - y2
        delta1 = y2 * (1 - y2) * e1
        e2 = w2 * delta1    # 1*4
        delta2 = y1 * (1 - y1) * e2     # 1*4
        w2 = w2 + alpha * delta1 * y1
        w1 = w1 + alpha * (iput[i].reshape(3, 1) @ delta2).T
    return w1, w2

for _ in range(8000):
    w1, w2 = backpropXOR(w1, w2, iput, oput)
v1 = iput @ w1.T    # 4*4
y1 = mySigmoid(v1)
v2 = y1 @ w2.reshape(hiden_node, 1) # 4*1
y2 = mySigmoid(v2)

print("最终结果为：",y2)
