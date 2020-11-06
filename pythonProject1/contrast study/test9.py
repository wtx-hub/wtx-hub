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



n_input = 1
n_step = 1
n_hidden = 1
batch_size = 1
learningrate=0.01

normaldata0=[]
normaldata1=[]
normaldata2=[]
normaldata3=[]
normaldata4=[]
normaldata5=[]
normaldata6=[]

for data00 in data0:
    normaldata0.append((data00-np.mean(data0))/np.std(data0))
for data11 in data1:
    normaldata1.append((data11-np.mean(data1))/np.std(data1))
for data22 in data2:
    normaldata2.append((data22-np.mean(data2))/np.std(data2))
for data33 in data3:
    normaldata3.append((data33-np.mean(data3))/np.std(data3))
for data44 in data4:
    normaldata4.append((data44-np.mean(data4))/np.std(data4))
for data55 in data5:
    normaldata5.append((data55-np.mean(data5))/np.std(data5))
for data66 in data6:
    normaldata6.append((data66-np.mean(data6))/np.std(data6))


# placeholder
x1 = tf.placeholder(tf.float32,[n_input,n_step])
x2 = tf.placeholder(tf.float32,[n_input,n_step])
x3 = tf.placeholder(tf.float32,[n_input,n_step])
x4 = tf.placeholder(tf.float32,[n_input,n_step])

def f(x):
    """
    定义激活函数f
    """
    return x
#定义常用激活函数
class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + tf.exp(-weighted_input))
    def backward(self, output):
        return output * (1 - output)
class TanhActivator(object):
    def forward(self, weighted_input):
        return 2.0 / (1.0 + tf.exp(-2 * weighted_input)) - 1.0
    def backward(self, output):
        return 1 - output * output

#定义lstm框架
class Lstm(object):
    def __init__(self,n_input,n_step,n_hidden):
        self.n_input = n_input
        self.n_step = n_step
        self.n_hidden = n_hidden
        # 门的激活函数
        self.gate_activator = SigmoidActivator()
        # 输出的激活函数
        self.output_activator = TanhActivator()
        # 当前时刻初始化为t0
        self.times = 0
        self.Wfh,self.Wfx,self.bf = self.init_weight_mat()
        # 输入门权重矩阵Wfh, Wfx, 偏置项bf
        self.Wih, self.Wix, self.bi = self.init_weight_mat()
        # 输出门权重矩阵Wfh, Wfx, 偏置项bf
        self.Woh, self.Wox, self.bo = self.init_weight_mat()
        # 单元状态权重矩阵Wfh, Wfx, 偏置项bf
        self.Wch, self.Wcx, self.bc = self.init_weight_mat()
        # 各个时刻的单元状态向量c
        self.c_list = self.init_state_vec()
        # 各个时刻的输出向量h
        self.h_list = self.init_state_vec()
        # 各个时刻的遗忘门f
        self.f_list = self.init_state_vec()
        # 各个时刻的输入门i
        self.i_list = self.init_state_vec()
        # 各个时刻的输出门o
        self.o_list = self.init_state_vec()
        # 各个时刻的即时状态c~
        self.ct_list = self.init_state_vec()
    def init_weight_mat(self):
        '''
        初始化权重矩阵
        '''
        Wh = tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden]))
        Wx = tf.Variable(tf.random_normal([self.n_hidden, self.n_input]))
        b = tf.Variable(tf.random_normal([self.n_hidden,self.n_input]))
        return Wh, Wx, b
    def init_state_vec(self):
        '''
        初始化保存状态的向量
        '''
        state_vec_list=[]
        state_vec_list.append([[0]])
        return state_vec_list
    def calc_gate(self,x,Wx,Wh,b,activator):
        '''
        计算门
        '''
        h=self.h_list[self.times - 1]  # 上次的LSTM输出
        net = tf.matmul(Wh, h) + tf.matmul(Wx, x) + b
        gate = activator.forward(net)
        return gate
    def forward(self, x):
        '''
        根据式1-式6进行前向计算
        '''
        self.times += 1
        # 遗忘门
        fg = self.calc_gate(x,self.Wfh,self.Wfx,self.bf,self.gate_activator)
        self.f_list.append(fg)
        # 输入门
        ig = self.calc_gate(x, self.Wih,self.Wix,self.bi, self.gate_activator)
        self.i_list.append(ig)
        # 输出门
        og = self.calc_gate(x, self.Woh,self.Wox,self.bo, self.gate_activator)
        self.o_list.append(og)
        # 即时状态
        ct = self.calc_gate(x, self.Wch,self.Wcx,self.bc, self.output_activator)
        self.ct_list.append(ct)
        # 单元状态
        c = fg * self.c_list[self.times - 1] + ig * ct
        self.c_list.append(c)
        # 输出
        h = og * self.output_activator.forward(c)
        self.h_list.append(h)
        return h

#定义mlp框架
class mlp(object):
    def __init__(self):
        self.weights1 = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
        # 偏置项初始化，并设置为共享变量
        self.bias1 = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
        # 权重向量初始化,并设置为共享变量
        self.weights2 = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
        # 偏置项初始化，并设置为共享变量
        self.bias2 = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
        # 权重向量初始化,并设置为共享变量
        self.weights3 = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
        # 偏置项初始化，并设置为共享变量
        self.bias3 = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
    def forward(self,x,y):
        # 公式
        result1 = f(tf.matmul(self.weights1, x) + self.bias1)
        # 公式
        result2 = f(tf.matmul(self.weights2, y) + self.bias2)
        # 公式
        result3 = f(tf.matmul(self.weights3, result1) + tf.matmul(self.weights3, result2) + self.bias3)
        return result3

# 定义计算图
lstm1 = Lstm(n_input, n_step, n_hidden)
lstm2 = Lstm(n_input, n_step, n_hidden)
mlp1 = mlp()
h1 = lstm1.forward(x1)
h2 = lstm2.forward(x2)
z1 = mlp1.forward(h1, h2)
h3 = lstm1.forward(x3)
h4 = lstm2.forward(x4)
z2 = mlp1.forward(h1, h2)
# 定义损失函数
loss =tf.square(tf.log(z1/z2))
# 形成计算图，将变量初始化，进行运算
train_op = tf.train.AdamOptimizer(learningrate).minimize(loss)
init_op = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())
def train():
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(40):
            sess.run(train_op, feed_dict={x1: [[normaldata0[i]]], x2: [[normaldata1[i]]], x3: [[normaldata2[i]]], x4: [[normaldata3[i]]]})
            # sess.run(saver)
        finalresult = []
        for i in range(0,30):
            temp1 = float((sess.run(z1, feed_dict={x1: [[normaldata4[i]]], x2: [[normaldata5[i]]]})))
            # temp2=temp1*np.std(data6)+np.mean(data6)
            # temp3=float(temp2)
            finalresult.append(temp1)
        print(finalresult)
        axis = []
        for i in range(1, 31):
            axis.append(i)
        # contrast = []
        # for i in range(0, 30):
        #     contrast.append(normaldata6[i])
        plt.figure()
        plt.plot(axis, finalresult)
        # plt.plot(axis, contrast)
        plt.grid(True)
        plt.show()
train()


