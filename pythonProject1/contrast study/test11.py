import tensorflow.compat.v1 as tf
import pandas as pd
tf.compat.v1.disable_eager_execution()
#设置GPU按需增长
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True#当使用GPU时候，Tensorflow运行自动慢慢达到最大GPU的内存
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess=tf.compat.v1.Session(config=config)

#取数据
df = pd.read_excel(r'workingD.xlsx', sheet_name=0)  # 取得是正调整样本
data0 = df.iloc[:, 0].values  # 柜位
data1 = df.iloc[:, 4].values  # 产消和
df1 = pd.read_excel(r'workingD.xlsx', sheet_name=1)  # 取得是负调整样本
data2 = df1.iloc[:, 0].values  # 柜位
data3 = df1.iloc[:, 4].values  # 产消和



#定义LSTM框架
def sim(a, b):
    # return tf.matmul(a,b)
    return tf.matmul(tf.transpose(a), b) / (tf.sqrt(tf.matmul(tf.transpose(a),a)) * tf.sqrt(tf.matmul(tf.transpose(b),b)))
def f(x):
    """
    定义激活函数f
    """
    return 1.0 / (1.0 + tf.exp(-x))
def Sigmoidforward(x):
    return 1.0 / (1.0 + tf.exp(-x))
def Tanhforward(x):
    return 2.0 / (1.0 + tf.exp(-2 * x)) - 1.0
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
def calc_gate(x,Wx,Wh,b,Sigmoidforward):
    '''
    计算门
    '''
    h=h_list[times - 1]  # 上次的LSTM输出
    net = tf.matmul(Wh, h) + tf.matmul(Wx, x) + b
    gate = Sigmoidforward(net)
    return gate


n_input = 1
n_step = 1
n_hidden = 1
batch_size = 1
lreaningrate=0.01
# 当前时刻初始化为t0
times = 0



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


#定义lstm前向计算公式
def lstm(x):
    # 遗忘门
    fg = calc_gate(x,Wfh,Wfx,bf,Sigmoidforward)
    f_list.append(fg)
    # 输入门
    ig = calc_gate(x,Wih,Wix,bi,Sigmoidforward)
    i_list.append(ig)
    # 输出门
    og = calc_gate(x,Woh,Wox,bo,Sigmoidforward)
    o_list.append(og)
    # 即时状态
    ct = calc_gate(x, Wch,Wcx,bc,Tanhforward)
    ct_list.append(ct)
    # 单元状态
    c = fg * c_list[times - 1] + ig * ct
    c_list.append(c)
    # 输出
    h = og * Sigmoidforward(c)
    h_list.append(h)
    return h

h1=lstm(x1)
h2=lstm(x2)

#定义MLP前向计算公式
def mlp(x1,x2):
    # 公式
    result1 = f(tf.matmul(weights1, x1) + bias1)
    # 公式
    result2 = f(tf.matmul(weights2, x2) + bias2)
    # 公式
    result3 = f(tf.matmul(weights3, result1) + tf.matmul(weights3, result2) + bias3)
    return result3

z1=mlp(h1,h2)


#计算损失函数
Prlist = []
Nrlist = []

def loss():
    loss_molecule_value=0
    temperature=1.0
    for i in range(47):
        for j in range(47):
            if i==j:
                    break
            else:
                loss_molecule_value+=(tf.exp(sim(Prlist[i],Prlist[j])))/temperature
    print(loss_molecule_value)

    loss_denominator_value=0
    temperature=1.0
    for i in range(43):
        for j in range(43):
            if i==j:
                break
            else:
                loss_denominator_value+=(tf.exp(sim(Nrlist[i],Nrlist[j])))/temperature
    print(loss_denominator_value)

    loss=-tf.log(loss_molecule_value/loss_denominator_value)
    return loss

def lstmmlprunning():
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(len(data0)):
            pred1=sess.run(z1, feed_dict={x1:[[data0[i]]],x2:[[data1[i]]]})
            Prlist.append(pred1)
        for i in range(len(data2)):
            pred2 = sess.run(z1, feed_dict={x1: [[data2[i]]], x2: [[data3[i]]]})
            Nrlist.append(pred2)
        losss = loss()
        train_op = tf.train.AdamOptimizer(lreaningrate).minimize(losss)
        saver = tf.train.Saver(tf.global_variables())
        sess.run(losss, train_op, saver)


lstmmlprunning()

# def trainrunning():
#     init_op = tf.global_variables_initializer()
#     losss = loss()
#     train_op = tf.train.AdamOptimizer(lreaningrate).minimize(losss)
#     saver = tf.train.Saver(tf.global_variables())
#     with tf.Session() as sess:
#         sess.run(init_op)
#         sess.run(losss,train_op,saver)
# trainrunning()



