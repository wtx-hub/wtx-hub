from __future__ import print_function
from functools import reduce
import pandas as pd
import numpy as np
import math
verification_length=1
input_size=1

df=pd.read_excel(r'workingC.xlsx')
data=df.iloc[:,0].values
train_x=[]
for i in range(len(data)):
    x = data[i:i + verification_length]
    train_x.append(x)
train_x = np.array(train_x)
train_x=train_x.reshape(-1,1)

data1=df.iloc[:,1].values
train_y=[]
for i in range(len(data1)):
    y=data1[i:i+verification_length]
    train_y.append(y)
train_y=np.array(train_y)
train_y=train_y.reshape(-1,1)

class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))
    def backward(self, output):
        return output * (1 - output)
class TanhActivator(object):
    def forward(self, weighted_input):
        return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0
    def backward(self, output):
        return 1 - output * output
class LstmLayer(object):
    def __init__(self, input_width, state_width,
                 learning_rate):
        self.input_width = input_width
        self.state_width = state_width
        self.learning_rate = learning_rate
        # 门的激活函数
        self.gate_activator = SigmoidActivator()
        # 输出的激活函数
        self.output_activator = TanhActivator()
        # 当前时刻初始化为t0
        self.times = 0
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
        # 遗忘门权重矩阵Wfh, Wfx, 偏置项bf
        self.Wf,  self.bf = (
            self.init_weight_mat())
        # 输入门权重矩阵Wfh, Wfx, 偏置项bf
        self.Wi, self.bi = (
            self.init_weight_mat())
        # 输出门权重矩阵Wfh, Wfx, 偏置项bf
        self.Wo, self.bo = (
            self.init_weight_mat())
        # 单元状态权重矩阵Wfh, Wfx, 偏置项bf
        self.Wc, self.bc = (
            self.init_weight_mat())
    def init_state_vec(self):
        '''
        初始化保存状态的向量
        '''
        state_vec_list = []
        state_vec_list.append(np.zeros(
            (self.state_width,self.input_width)))
        return state_vec_list
    def init_weight_mat(self):
        '''
        初始化权重矩阵
        '''
        Wf = np.random.uniform(-1e-4, 1e-4,
            (2*self.input_width, self.input_width))
        b = np.zeros((self.state_width, self.input_width))
        return Wf, b
    def calc_gate(self, x, Wf, b, activator):
        '''
        计算门
        '''
        h = self.h_list[self.times - 1]  # 上次的LSTM输出
        c1=np.hstack((h,x))
        net = np.dot(c1 , Wf) + b
        gate = activator.forward(net)
        return gate
    def forward(self, x):
        '''
        根据式1-式6进行前向计算
        '''
        self.times += 1
        # 遗忘门
        fg = self.calc_gate(x, self.Wf,
                            self.bf, self.gate_activator)
        self.f_list.append(fg)
        # 输入门
        ig = self.calc_gate(x, self.Wi,
                            self.bi, self.gate_activator)
        self.i_list.append(ig)
        # 输出门
        og = self.calc_gate(x, self.Wo,
                            self.bo, self.gate_activator)
        self.o_list.append(og)
        # 即时状态
        ct = self.calc_gate(x, self.Wc,
                            self.bc, self.output_activator)
        self.ct_list.append(ct)
        # 单元状态
        c = fg * self.c_list[self.times - 1] + ig * ct
        self.c_list.append(c)
        # 输出
        h = og * self.output_activator.forward(c)
        self.h_list.append(h)

model=LstmLayer(1,90,0.01)
model.forward(train_x)
# print(model.h_list[model.times])
a=model.h_list[model.times]
model1=LstmLayer(1,90,0.01)
model1.forward(train_y)
# print(model1.h_list[model1.times])
b=model1.h_list[model1.times]
print(a)
print(b)




class VectorOp(object):
    """
    实现向量计算操作
    """
    @staticmethod
    def dot(x, y):
        """
        计算两个向量x和y的内积
        """
        # 首先把x[x1,x2,x3...]和y[y1,y2,y3,...]按元素相乘
        # 变成[x1*y1, x2*y2, x3*y3]
        # 然后利用reduce求和
        return reduce(lambda a, b: a + b, VectorOp.element_multiply(x, y), 0.0)

    @staticmethod
    def element_multiply(x, y):
        """
        将两个向量x和y按元素相乘
        """
        # 首先把x[x1,x2,x3...]和y[y1,y2,y3,...]打包在一起
        # 变成[(x1,y1),(x2,y2),(x3,y3),...]
        # 然后利用map函数计算[x1*y1, x2*y2, x3*y3]
        return list(map(lambda x_y: x_y[0] * x_y[1], zip(x, y)))

    @staticmethod
    def element_add(x, y):
        """
        将两个向量x和y按元素相加
        """
        # 首先把x[x1,x2,x3...]和y[y1,y2,y3,...]打包在一起
        # 变成[(x1,y1),(x2,y2),(x3,y3),...]
        # 然后利用map函数计算[x1+y1, x2+y2, x3+y3]
        return list(map(lambda x_y: x_y[0] + x_y[1], zip(x, y)))

    @staticmethod
    def scala_multiply(v, s):
        """
        将向量v中的每个元素和标量s相乘
        """
        return map(lambda e: e * s, v)


class Perceptron(object):
    def __init__(self, input_num, activator):
        """
        初始化感知器，设置输入参数的个数，以及激活函数。
        激活函数的类型为double -> double
        """
        self.activator = activator
        # 权重向量初始化为0
        self.weights = [0.0] * input_num
        # 偏置项初始化为0
        self.bias = 0.0

    def __str__(self):
        """
        打印学习到的权重、偏置项
        """
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        """
        输入向量，输出感知器的计算结果
        """
        # 计算向量input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]的内积
        # 然后加上bias
        return self.activator(
            VectorOp.dot(input_vec, self.weights) + self.bias)




def f(x):
    """
    定义激活函数f
    """
    return 1.0 / (1.0 + np.exp(-x))

model2=Perceptron(1,f)
c=model2.predict(a)
model3=Perceptron(1,f)
d=model3.predict(b)
print(c)
print(d)
def prod(x):
    return math.exp(x)
loss=-math.log10(reduce(prod,[]))