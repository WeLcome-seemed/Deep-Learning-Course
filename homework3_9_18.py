import numpy as np

iput = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
oput = np.array([0, 1, 1, 0])
w = 2 * np.random.rand(3,1) - 1
alpha = 0.8

def mySigmoid(a):
    return 1/(1 + np.exp(-a))

def mySGD(iMatrix, oMatrix, w):
    for i in range(0, 4):
        v = iMatrix[i] @ w 
        y = mySigmoid(v)
        e = oMatrix[i] - y
        delta = y * (1 - y) * e
        w = w + alpha * delta * (iMatrix[i].reshape(3, 1))
    return w

for _ in range(4000):  
    w = mySGD(iput, oput, w)
o_SGD = iput @ w
o_SGD = mySigmoid(o_SGD)

print("最终权重为：\n", w)
print("四个样本的输出为：\n", o_SGD)
