import tensorflow.compat.v1 as tf
import pandas as pd
tf.compat.v1.disable_eager_execution()

def f(x):
    """
    定义激活函数f
    """
    return 1.0 / (1.0 + tf.exp(-x))

n_input = 1
n_step = 1
n_hidden = 1
batch_size = 1
lreaningrate=0.01

#mlp处理
rate=0.1
iteration=100
"""
初始化感知器，设置输入参数的个数，以及激活函数。
激活函数的类型
"""
x1 = tf.placeholder(tf.float32,[n_input,n_step])
x2 = tf.placeholder(tf.float32,[n_input,n_step])

# 权重向量初始化,并设置为共享变量
weights1=tf.Variable(tf.random_normal([n_hidden, n_hidden]))
# 偏置项初始化，并设置为共享变量
bias1=tf.Variable(tf.random_normal([n_hidden, n_hidden]))
#公式
result1=f(tf.matmul(weights1,x1)+bias1)


# 权重向量初始化,并设置为共享变量
weights2=tf.Variable(tf.random_normal([n_hidden, n_hidden]))
# 偏置项初始化，并设置为共享变量
bias2=tf.Variable(tf.random_normal([n_hidden, n_hidden]))
#公式
result2=f(tf.matmul(weights2,x2)+bias2)

# 权重向量初始化,并设置为共享变量
weights3=tf.Variable(tf.random_normal([n_hidden, n_hidden]))
# 偏置项初始化，并设置为共享变量
bias3=tf.Variable(tf.random_normal([n_hidden, n_hidden]))
#公式
result3=f(tf.matmul(weights3,result1)+tf.matmul(weights3,result2)+bias3)
#最终公式，输出经过LSTM和两个BP神经网络的表征值，表征值是由产消和与柜位两个元素决定的
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(result3, feed_dict={x1:[[3.0]],x2:[[2.0]]}))



# #计算损失函数
# def sim(a, b):
#     return np.dot(a.T, b) / (np.sqrt(a.T * a) * np.sqrt(b.T * b))
# def loss():
#     loss_molecule_value=0
#     temperature=1.0
#     for i in range(model0.times):
#         for j in range(model0.times):
#             if i==j:
#                 break
#             else:
#                 loss_molecule_value+=(np.exp(sim(Prlist[i],Prlist[j])))/temperature
#     print(loss_molecule_value)
#
#     loss_denominator_value=0
#     temperature=1.0
#     for i in range(model2.times):
#         for j in range(model2.times):
#             if i==j:
#                 break
#             else:
#                 loss_denominator_value+=(np.exp(sim(Nrlist[i],Nrlist[j])))/temperature
#     print(loss_denominator_value)
#
#     loss=-np.log(loss_molecule_value/loss_denominator_value)
#     return loss