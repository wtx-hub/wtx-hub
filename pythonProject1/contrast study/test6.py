from theano import *
import numpy as np
import theano.tensor as T

def f(x):
    """
    定义激活函数f
    """
    return 1.0 / (1.0 + np.exp(-x))

target=0.8
rate=0.1
iteration=100
"""
初始化感知器，设置输入参数的个数，以及激活函数。
激活函数的类型
"""
x =T.dscalar()
# 权重向量初始化,并设置为共享变量
weights=theano.shared(0.5)
# 偏置项初始化，并设置为共享变量
bias=theano.shared(0.5)
#公式
result=f(weights * x + bias)
loss=target-result
dwloss=T.grad(loss,weights)
dbloss=T.grad(loss,bias)
f= theano.function([x],[result,loss,dwloss,dbloss],updates=[(weights,weights-rate*dwloss),(bias,bias-rate*dbloss)])
for i in range(iteration):
    print(f(2))
    print(weights.get_value())

    print(bias.get_value())
# y2 = f(weights,bias)
# loss=target-y2
# y3= theano.function([y2], loss)
# loss=y3(y2)
# print(loss)

# # 更新weights
# weights=weights-rate *dwloss
# # 更新bias
# bias=bias-rate *dbloss
# a0 = weights * x + bias
# a1 = f(a0)
# loss=target-a1
# print(loss)
# for i in range(iteration):
#     # dwloss=theano.grad(loss,a1)*theano.grad(a1,a0)*theano.grad(a0,weights)
#     # dbloss=theano.grad(loss,a1)*theano.grad(a1,a0)*theano.grad(a0,bias)
#     dwloss=theano.grad(loss,weights)
#     dbloss=theano.grad(loss,bias)
#     # 更新weights
#     weights=weights-rate *dwloss
#     # 更新bias
#     bias=bias-rate *dbloss

# print(a1)
    


