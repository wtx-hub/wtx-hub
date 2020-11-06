from __future__ import print_function
from functools import reduce
import pandas as pd
import numpy as np
import math
batch_size=1
n_input=1
n_steps=90

def f(x):
    """
    定义激活函数f
    """
    return max([0,x])

#定义LSTM框架
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
        self.Wfh,self.Wfx,self.bf = (self.init_weight_mat())
        # 输入门权重矩阵Wfh, Wfx, 偏置项bf
        self.Wih,self.Wix,self.bi = (self.init_weight_mat())
        # 输出门权重矩阵Wfh, Wfx, 偏置项bf
        self.Woh,self.Wox,self.bo = (self.init_weight_mat())
        # 单元状态权重矩阵Wfh, Wfx, 偏置项bf
        self.Wch,self.Wcx,self.bc = (self.init_weight_mat())
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
        Wh = np.random.uniform(-1e-4, 1e-4,(self.state_width, self.state_width))
        Wx = np.random.uniform(-1e-4, 1e-4,(self.state_width, self.input_width))
        b = np.zeros((self.state_width, self.input_width))
        return Wh,Wx,b
    def calc_gate(self,x,Wx,Wh,b,activator):
        '''
        计算门
        '''
        h = self.h_list[self.times - 1]  # 上次的LSTM输出
        # c1=np.hstack((h,x))
        net = np.dot(Wh,h)+np.dot(Wx,x) + b
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

#取数据并进行LSTM处理
df=pd.read_excel(r'workingD.xlsx',sheet_name=0)#取得是正调整样本

model0=LstmLayer(1,1,0.01)
data0=df.iloc[:,0].values#柜位
for data00 in data0:
    model0.forward(data00)

model1=LstmLayer(1,1,0.01)
data1=df.iloc[:,4].values#产消和
for data11 in data1:
    model1.forward(data11)

df1 = pd.read_excel(r'workingD.xlsx', sheet_name=1)  # 取得是负调整样本

model2 = LstmLayer(1, 1, 0.01)
data2 = df1.iloc[:, 0].values  # 柜位
for data22 in data2:
    model2.forward(data22)

model3 = LstmLayer(1, 1, 0.01)
data3 = df1.iloc[:, 4].values  # 产消和
for data33 in data3:
    model3.forward(data33)





print(model0.h_list[model0.times])
print(model1.h_list[model1.times])
print(model2.h_list[model2.times])
print(model3.h_list[model3.times])




#定义MLP框架
class VectorOp(object):
    """
    实现向量计算操作
    """
    @staticmethod
    def dot(x, y):
        # 首先把x[x1,x2,x3...]和y[y1,y2,y3,...]按元素相乘变成[x1*y1, x2*y2, x3*y3]
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
        self.weights = [10.0] * input_num
        # 偏置项初始化为0
        self.bias = 0.1

    def __str__(self):
        """
        打印学习到的权重、偏置项
        """
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)
    def _update_weights(self, input_vec, output, label, rate):
        """
        按照感知器规则更新权重
        """
        # 首先计算本次更新的delta
        # 然后把input_vec[x1,x2,x3,...]向量中的每个值乘上delta，得到每个权重更新
        # 最后再把权重更新按元素加到原先的weights[w1,w2,w3,...]上
        delta = label - output
        self.weights = VectorOp.element_add(
            self.weights, VectorOp.scala_multiply(input_vec, rate * delta))
        # 更新bias
        self.bias += rate * delta
    def _one_iteration(self, input_vecs, labels, rate):
        """
        一次迭代，把所有的训练数据过一遍
        """
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        samples = zip(input_vecs, labels)
        # 对每个样本，按照感知器规则更新权重
        for (input_vec, label) in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec, output, label, rate)
    def train(self, input_vecs, labels, iteration, rate):
        """
        输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
        """
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)
    def predict(self, input_vec):
        """
        输入向量，输出感知器的计算结果
        """
        # 计算向量input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]的内积
        # 然后加上bias
        return self.activator(
            VectorOp.dot(input_vec, self.weights) + self.bias)

#将正调节样本的柜位和产销差数据进行可学习的MLP神经网络处理
def Pr(x,y):
    model4=Perceptron(1,f)
    # g1=model2.predict(model0.h_list[model0.times])
    g1=model4.predict(x)
    # g2=model2.predict(model1.h_list[model1.times])
    g2=model4.predict(y)
    model5=Perceptron(2,f)
    g12=model5.predict([g1,g2])
    return g12
Prlist=[]
for i in range(model0.times):
    g=Pr(model0.h_list[i],model1.h_list[i])
    Prlist.append(g)
print(Prlist[0])#得到的是正调节样本的表征值

#将负调节样本的柜位和产销差数据进行可学习的MLP神经网络处理
def Nr(x,y):
    model6=Perceptron(1,f)
    # g1=model2.predict(model0.h_list[model0.times])
    g1=model6.predict(x)
    # g2=model2.predict(model1.h_list[model1.times])
    g2=model6.predict(y)
    model7=Perceptron(2,f)
    g12=model7.predict([g1,g2])
    return g12
Nrlist=[]
for i in range(model2.times):
    gg=Nr(model2.h_list[i],model3.h_list[i])
    Nrlist.append(gg)
print(Nrlist[0])#得到的是负调节样本的表征值

#计算损失函数
def sim(a, b):
    return np.dot(a.T, b) / (np.sqrt(a.T * a) * np.sqrt(b.T * b))
def loss():
    loss_molecule_value=0
    temperature=1.0
    for i in range(model0.times):
        for j in range(model0.times):
            if i==j:
                break
            else:
                loss_molecule_value+=(np.exp(sim(Prlist[i],Prlist[j])))/temperature
    print(loss_molecule_value)

    loss_denominator_value=0
    temperature=1.0
    for i in range(model2.times):
        for j in range(model2.times):
            if i==j:
                break
            else:
                loss_denominator_value+=(np.exp(sim(Nrlist[i],Nrlist[j])))/temperature
    print(loss_denominator_value)

    loss=-np.log(loss_molecule_value/loss_denominator_value)
    return loss
pp=loss()
print(pp)
