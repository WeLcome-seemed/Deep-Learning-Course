import numpy as np
import matplotlib.pyplot as plt

iput = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
oput = np.array([0, 0, 1, 1])
w = 2 * np.random.rand(3,1) - 1
alpha = 0.8

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
    e = oMatrix.reshape(4, 1) - mySigmoid(iMatrix @ w)
    e = np.sum(e * e)
    return w,e

#batch
def mybatch(iMatrix, oMatrix, w):
    delta_w = 0
    for i in range(0, 4):
        v = iMatrix[i] @ w 
        y = mySigmoid(v)
        e = oMatrix[i] - y
        delta = y * (1 - y) * e
        delta_w = delta_w + alpha * delta * (iMatrix[i].reshape(3, 1))
    e = oMatrix.reshape(4, 1) - mySigmoid(iMatrix @ w)
    e = np.sum(e * e)
    return w + delta_w/4, e

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
    e = oMatrix.reshape(4, 1) - mySigmoid(iMatrix @ w)
    e = np.sum(e * e)
    return w, e

w_SGD = w
err_SGD = [ 0 for _ in range(1000)]
for i in range(1000):  
    w_SGD, e_SGD = mySGD(iput, oput, w_SGD)
    err_SGD[i] = e_SGD

w_batch = w
err_batch = [ 0 for _ in range(1000)]
for i in range(1000):
    w_batch, e_batch = mybatch(iput, oput, w_batch)
    err_batch[i] = e_batch

w_minibatch = w
err_minibatch = [ 0 for _ in range(1000)]
for i in range(1000):
    w_minibatch, e_minibatch = myminibatch(iput, oput, w_minibatch)
    err_minibatch[i] = e_minibatch


#作图
epoch = list(range(1, 1001))
plt.xlabel("Epoch")
plt.ylabel("Squared Error")
plt.title("Training Error Curves")
plt.plot(epoch, err_SGD, label = "SGD")
plt.plot(epoch, err_batch, label = "批训练")
plt.plot(epoch, err_minibatch, label = "小批训练")
plt.show()





    

