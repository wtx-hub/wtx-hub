import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()
#设置GPU按需增长
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True#当使用GPU时候，Tensorflow运行自动慢慢达到最大GPU的内存
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess=tf.compat.v1.Session(config=config)


def f(x):
    """
    定义激活函数f
    """
    return x
def tanh(x):
    return (tf.exp(x)-tf.exp(-x))/(tf.exp(x)+tf.exp(-x))
def Sigmoidforward(x):
    return 1.0 / (1.0 + tf.exp(-x))
def Tanhforward(x):
    return (2.0 / (1.0 + tf.exp(-2 * x))) - 1.0
def normalizedata(x):
    return (x-np.mean(x))/np.std(x)


#取数据
df = pd.read_excel(r'workingD.xlsx', sheet_name=0)  # 取得是正调整样本
data0 = df.iloc[:, 0].values  # 柜位
data1 = df.iloc[:, 4].values  # 产消和
df1 = pd.read_excel(r'workingD.xlsx', sheet_name=1)  # 取得是负调整样本
data2 = df1.iloc[:, 0].values  # 柜位
data3 = df1.iloc[:, 4].values  # 产消和
df2 = pd.read_excel(r'workingC.xlsx', sheet_name=0)  # 取得是全部样本
data4 = df2.iloc[:, 0].values  # 柜位
data5 = df2.iloc[:, 1].values  # 产消和
data6 = df2.iloc[:, 4].values  # 调整量


#标准化

# normaldata0=[]
# normaldata1=[]
# normaldata2=[]
# normaldata3=[]
normaldata4=[]
normaldata5=[]
normaldata6=[]
# for data00 in data0:
#     normaldata0.append((data00-np.mean(data0))/np.std(data0))
# for data11 in data1:
#     normaldata1.append((data11-np.mean(data1))/np.std(data1))
# for data22 in data2:
#     normaldata2.append((data22-np.mean(data2))/np.std(data2))
# for data33 in data3:
#     normaldata3.append((data33-np.mean(data3))/np.std(data3))
for data44 in data4:
    normaldata4.append((data44-np.mean(data4))/np.std(data4))

for data55 in data5:
    normaldata5.append((data55-np.mean(data5))/np.std(data5))

for data66 in data6:
    normaldata6.append((data66-np.mean(data6))/np.std(data6))


# print(normaldata4)
# print(normaldata5)
# print(normaldata6)
#定义LSTM框架

def init_weight_mat():
    '''
    初始化权重矩阵
    '''
    Wh = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
    Wx = tf.Variable(tf.random_normal([n_hidden, n_input]))
    b = tf.Variable(tf.random_normal([n_hidden,n_input]))
    return Wh, Wx, b
def init_state_vec():
    '''
    初始化保存状态的向量
    '''
    state_vec_list=[[[0]]]
    return state_vec_list
def calc_gate1(x,Wx,Wh,b,Sigmoidforward):
    '''
    计算门
    '''
    h=h_list[times1 - 1]  # 上次的LSTM输出
    net = tf.matmul(Wh, h) + tf.matmul(Wx, x) + b
    gate = Sigmoidforward(net)
    return gate
def calc_gate2(x,Wx,Wh,b,Sigmoidforward):
    '''
    计算门
    '''
    h=h_list[times2 - 1]  # 上次的LSTM输出
    net = tf.matmul(Wh, h) + tf.matmul(Wx, x) + b
    gate = Sigmoidforward(net)
    return gate


n_input = 1
n_step = 1
n_hidden = 1
batch_size = 1
learningrate=0.001
# 当前时刻初始化为t0
times1 = 0
times2 = 0


#权重初始化，并设置为共享变量(lstm)
Wfh,Wfx,bf = init_weight_mat()
# 输入门权重矩阵Wfh, Wfx, 偏置项bf
Wih, Wix, bi = init_weight_mat()
# 输出门权重矩阵Wfh, Wfx, 偏置项bf
Woh, Wox, bo = init_weight_mat()
# 单元状态权重矩阵Wfh, Wfx, 偏置项bf
Wch, Wcx, bc = init_weight_mat()
# 各个时刻的单元状态向量c
c_list = init_state_vec()
# 各个时刻的输出向量h
h_list = init_state_vec()
# 各个时刻的遗忘门f
f_list = init_state_vec()
# 各个时刻的输入门i
i_list = init_state_vec()
# 各个时刻的输出门o
o_list = init_state_vec()
# 各个时刻的即时状态c~
ct_list = init_state_vec()

#权重初始化，并设置为共享变量(lstm)
Wfh1,Wfx1,bf1 = init_weight_mat()
# 输入门权重矩阵Wfh, Wfx, 偏置项bf
Wih1, Wix1, bi1 = init_weight_mat()
# 输出门权重矩阵Wfh, Wfx, 偏置项bf
Woh1, Wox1, bo1 = init_weight_mat()
# 单元状态权重矩阵Wfh, Wfx, 偏置项bf
Wch1, Wcx1, bc1 = init_weight_mat()
# 各个时刻的单元状态向量c
c_list1 = init_state_vec()
# 各个时刻的输出向量h
h_list1 = init_state_vec()
# 各个时刻的遗忘门f
f_list1 = init_state_vec()
# 各个时刻的输入门i
i_list1 = init_state_vec()
# 各个时刻的输出门o
o_list1 = init_state_vec()
# 各个时刻的即时状态c~
ct_list1 = init_state_vec()

# 权重向量初始化,并设置为共享变量(MLP)
weights1 = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
# 偏置项初始化，并设置为共享变量
bias1 = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
# 权重向量初始化,并设置为共享变量
weights2 = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
# 偏置项初始化，并设置为共享变量
bias2 = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
# 权重向量初始化,并设置为共享变量
weights3 = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
# 偏置项初始化，并设置为共享变量
bias3 = tf.Variable(tf.random_normal([n_hidden, n_hidden]))

# placeholder
x1 = tf.placeholder(tf.float32,[n_input,n_step])
x2 = tf.placeholder(tf.float32,[n_input,n_step])
x3 = tf.placeholder(tf.float32,[n_input,n_step])



#定义lstm前向计算公式
def lstm(x):
    global times1
    times1+=1
    # 遗忘门
    fg = calc_gate1(x,Wfh,Wfx,bf,Sigmoidforward)
    f_list.append(fg)
    # 输入门
    ig = calc_gate1(x,Wih,Wix,bi,Sigmoidforward)
    i_list.append(ig)
    # 输出门
    og = calc_gate1(x,Woh,Wox,bo,Sigmoidforward)
    o_list.append(og)
    # 即时状态
    ct = calc_gate1(x, Wch,Wcx,bc,Tanhforward)
    ct_list.append(ct)
    # 单元状态
    c = fg * c_list[times1 - 1] + ig * ct
    c_list.append(c)
    # 输出
    h = og * Sigmoidforward(c)
    h_list.append(h)
    return h
def lstm1(x):
    global times2
    times2+=1
    # 遗忘门
    fg1 = calc_gate2(x,Wfh1,Wfx1,bf1,Sigmoidforward)
    f_list1.append(fg1)
    # 输入门
    ig1 = calc_gate2(x,Wih1,Wix1,bi1,Sigmoidforward)
    i_list1.append(ig1)
    # 输出门
    og1 = calc_gate2(x,Woh1,Wox1,bo1,Sigmoidforward)
    o_list1.append(og1)
    # 即时状态
    ct1 = calc_gate2(x, Wch1,Wcx1,bc1,Tanhforward)
    ct_list1.append(ct1)
    # 单元状态
    c1 = fg1 * c_list1[times2 - 1] + ig1 * ct1
    c_list1.append(c1)
    # 输出
    h1 = og1 * Sigmoidforward(c1)
    h_list1.append(h1)
    return h1


#定义MLP前向计算公式
def mlp(x,y):
    # 公式
    result1 = f(tf.matmul(weights1, x) + bias1)
    # 公式
    result2 = f(tf.matmul(weights2, y) + bias2)
    # 公式
    result3 = f(tf.matmul(weights3, result1) + tf.matmul(weights3, result2) + bias3)
    return result3

h1=lstm(x1)
h2=lstm1(x2)
z1=mlp(h1, h2)
loss=tf.square(x3-z1)
train_op = tf.train.AdamOptimizer(learningrate).minimize(loss)
init_op = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())

def lstmmlprunning():
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(80):
            sess.run(train_op,feed_dict={x1:[[normaldata4[i]]],x2:[[normaldata5[i]]],x3:[[normaldata6[i]]]})
        finalresult=[]
        for i in range(80,90):
            temp1=float((sess.run(z1, feed_dict={x1: [[normaldata4[i]]],x2:[[normaldata5[i]]]})))
            # temp2=temp1*np.std(data6)+np.mean(data6)
            # temp3=float(temp2)
            finalresult.append(temp1)
        print(finalresult)
        axis=[]
        for i in range(1,11):
            axis.append(i)
        contrast=[]
        for i in range(80,90):
            contrast.append(normaldata6[i])
        plt.figure()
        plt.plot(axis,finalresult)
        plt.plot(axis,contrast)
        plt.grid(True)
        plt.show()



lstmmlprunning()



