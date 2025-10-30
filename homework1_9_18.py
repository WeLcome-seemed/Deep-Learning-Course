import numpy as np

iput = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
oput = np.array([0, 0, 1, 1])
w = 2 * np.random.rand(3,1) - 1
alpha = 0.8
epoch_SGD = 1000
epoch_batch = 4000
epoch_minibatch = 2000

def mySigmoid(a):
    return 1/(1 + np.exp(-a))

#SGD
def mySGD(iMatrix, oMatrix, w):
    for i in range(0, 4):
        v = iMatrix[i] @ w 
        y = mySigmoid(v)
        e = oMatrix[i] - y
        delta = y * (1 - y) * e
        w = w + alpha * delta * (iMatrix[i].reshape(3, 1))
    return w

#batch
def mybatch(iMatrix, oMatrix, w):
    delta_w = 0
    for i in range(0, 4):
        v = iMatrix[i] @ w 
        y = mySigmoid(v)
        e = oMatrix[i] - y
        delta = y * (1 - y) * e
        delta_w = delta_w + alpha * delta * (iMatrix[i].reshape(3, 1))
    return w + delta_w/4

#minibatch
def myminibatch(iMatrix, oMatrix, w):
    delta_w = 0
    for i in range(0, 4):
        v = iMatrix[i] @ w 
        y = mySigmoid(v)
        e = oMatrix[i] - y
        delta = y * (1 - y) * e
        delta_w = delta_w + alpha * delta * (iMatrix[i].reshape(3, 1))
        if (i == 1) or (i == 3):
            w = w + delta_w/2
            delta_w = 0
    return w

w_SGD = w
for _ in range(epoch_SGD):  
    w_SGD = mySGD(iput, oput, w_SGD)
o_SGD = iput @ w_SGD
o_SGD = mySigmoid(o_SGD)

w_batch = w
for _ in range(epoch_batch):
    w_batch = mybatch(iput, oput, w_batch)
o_batch = iput @ w_batch
o_batch = mySigmoid(o_batch)

w_minibatch = w
for _ in range(epoch_minibatch):
    w_minibatch = myminibatch(iput, oput, w_minibatch)
o_minibatch = iput @ w_minibatch
o_minibatch = mySigmoid(o_minibatch)

print("SGD的最终权重为：\n",w_SGD)
print("SGD的最终结果为：\n",o_SGD)
print("批量训练的最终权重为：\n",w_SGD)
print("批量训练的最终结果为：\n",o_SGD)
print("小批量的最终权重为：\n",w_SGD)
print("小批量的最终结果为：\n",o_SGD)




    

