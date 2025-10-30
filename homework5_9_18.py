#训练浅层NN解决XOR问题(1隐藏层)，改变隐藏层节点数
#如何避免不收敛？
#合理设置步长alpha，以及合理初始化权重等
import numpy as np

iput = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
oput = np.array([0, 1, 1, 0])
hiden_node3 = 3
hiden_node5 = 5
hiden_node2 = 2
w13 = 2 * np.random.rand(hiden_node3, 3) - 1
w23 = 2 * np.random.rand(1, hiden_node3) - 1
w15 = 2 * np.random.rand(hiden_node5, 3) - 1
w25 = 2 * np.random.rand(1, hiden_node5) - 1
w12 = 2 * np.random.rand(hiden_node2, 3) - 1
w22 = 2 * np.random.rand(1, hiden_node2) - 1
alpha = 0.9

def mySigmoid(a):
    return 1/(1 + np.exp(-a))

def backpropXOR(w1, w2, iput, oput, hiden_node):
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

for _ in range(4000):
    w13, w23 = backpropXOR(w13, w23, iput, oput, hiden_node3)
    w15, w25 = backpropXOR(w15, w25, iput, oput, hiden_node5)
    w12, w22 = backpropXOR(w12, w22, iput, oput, hiden_node2)
v13 = iput @ w13.T    # 4*4
y13 = mySigmoid(v13)
v23 = y13 @ w23.reshape(hiden_node3, 1) # 4*1
y23 = mySigmoid(v23)
v15 = iput @ w15.T   
y15 = mySigmoid(v15)
v25 = y15 @ w25.reshape(hiden_node5, 1) 
y25 = mySigmoid(v25)
v12 = iput @ w12.T    
y12 = mySigmoid(v12)
v22 = y12 @ w22.reshape(hiden_node2, 1) 
y22 = mySigmoid(v22)

print("训练4000轮后3节点最终结果为：\n",y23)
print("训练4000轮后5节点最终结果为：\n",y25)
print("训练4000轮后2节点最终结果为：\n",y22)