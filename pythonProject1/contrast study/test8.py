from __future__ import print_function
from functools import reduce
import pandas as pd
import numpy as np
import math
from theano import *
import theano.tensor as T

batch_size=1
n_input=1
n_steps=90
loss_molecule_value=0
loss_denominator_value=0
temperature=1.0

def f(x):
    """
    定义激活函数f
    """
    return 1.0 / (1.0 + np.exp(-x))
def sim(a, b):
    return np.dot(a.T, b) / (np.sqrt(a.T * a) * np.sqrt(b.T * b))

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


    def backward(self, x, delta_h, activator):
        '''
        实现LSTM训练算法
        '''
        self.calc_delta(delta_h, activator)
        self.calc_gradient(x)
        self.update()


    def calc_delta(self, delta_h, activator):
        # 初始化各个时刻的误差项
        self.delta_h_list = self.init_delta()  # 输出误差项
        self.delta_o_list = self.init_delta()  # 输出门误差项
        self.delta_i_list = self.init_delta()  # 输入门误差项
        self.delta_f_list = self.init_delta()  # 遗忘门误差项
        self.delta_ct_list = self.init_delta()  # 即时输出误差项

        # 保存从上一层传递下来的当前时刻的误差项
        self.delta_h_list[-1] = delta_h

        # 迭代计算每个时刻的误差项
        for k in range(self.times,0,-1):
            self.calc_delta_k(k)


    def init_delta(self):
        '''
        初始化误差项
        '''
        delta_list=[]
        for i in range(self.times + 1):
            delta_list.append(np.zeros(
                (self.state_width, 1)))
        return delta_list


    def calc_delta_k(self, k):
        '''
        根据k时刻的delta_h，计算k时刻的delta_f、
        delta_i、delta_o、delta_ct，以及k-1时刻的delta_h
        '''
        # 获得k时刻前向计算的值
        ig = self.i_list[k]
        og = self.o_list[k]
        fg = self.f_list[k]
        ct = self.ct_list[k]
        c = self.c_list[k]
        c_prev = self.c_list[k - 1]
        tanh_c = self.output_activator.forward(c)
        delta_k = self.delta_h_list[k]

        # 根据式9计算delta_o
        delta_o = (delta_k * tanh_c *
                   self.gate_activator.backward(og))
        delta_f = (delta_k * og *
                   (1 - tanh_c * tanh_c) * c_prev *
                   self.gate_activator.backward(fg))
        delta_i = (delta_k * og *
                   (1 - tanh_c * tanh_c) * ct *
                   self.gate_activator.backward(ig))
        delta_ct = (delta_k * og *
                    (1 - tanh_c * tanh_c) * ig *
                    self.output_activator.backward(ct))
        delta_h_prev = (
                np.dot(delta_o.transpose(), self.Woh) +
                np.dot(delta_i.transpose(), self.Wih) +
                np.dot(delta_f.transpose(), self.Wfh) +
                np.dot(delta_ct.transpose(), self.Wch)
        ).transpose()

        # 保存全部delta值
        self.delta_h_list[k - 1] = delta_h_prev
        self.delta_f_list[k] = delta_f
        self.delta_i_list[k] = delta_i
        self.delta_o_list[k] = delta_o
        self.delta_ct_list[k] = delta_ct


    def calc_gradient(self, x):
        # 初始化遗忘门权重梯度矩阵和偏置项
        self.Wfh_grad, self.Wfx_grad, self.bf_grad = (
            self.init_weight_gradient_mat())
        # 初始化输入门权重梯度矩阵和偏置项
        self.Wih_grad, self.Wix_grad, self.bi_grad = (
            self.init_weight_gradient_mat())
        # 初始化输出门权重梯度矩阵和偏置项
        self.Woh_grad, self.Wox_grad, self.bo_grad = (
            self.init_weight_gradient_mat())
        # 初始化单元状态权重梯度矩阵和偏置项
        self.Wch_grad, self.Wcx_grad, self.bc_grad = (
            self.init_weight_gradient_mat())

        # 计算对上一次输出h的权重梯度
        for t in range(self.times, 0, -1):
            # 计算各个时刻的梯度
            (Wfh_grad, bf_grad,
             Wih_grad, bi_grad,
             Woh_grad, bo_grad,
             Wch_grad, bc_grad) = (
                self.calc_gradient_t(t))
            # 实际梯度是各时刻梯度之和
            self.Wfh_grad += Wfh_grad
            self.bf_grad += bf_grad
            self.Wih_grad += Wih_grad
            self.bi_grad += bi_grad
            self.Woh_grad += Woh_grad
            self.bo_grad += bo_grad
            self.Wch_grad += Wch_grad
            self.bc_grad += bc_grad

        # 计算对本次输入x的权重梯度
        xt = x.transpose()
        self.Wfx_grad = np.dot(self.delta_f_list[-1], xt)
        self.Wix_grad = np.dot(self.delta_i_list[-1], xt)
        self.Wox_grad = np.dot(self.delta_o_list[-1], xt)
        self.Wcx_grad = np.dot(self.delta_ct_list[-1], xt)


    def init_weight_gradient_mat(self):
        '''
        初始化权重矩阵
        '''
        Wh_grad = np.zeros((self.state_width,
                            self.state_width))
        Wx_grad = np.zeros((self.state_width,
                            self.input_width))
        b_grad = np.zeros((self.state_width, 1))
        return Wh_grad, Wx_grad, b_grad


    def calc_gradient_t(self, t):
        '''
        计算每个时刻t权重的梯度
        '''
        h_prev = self.h_list[t - 1].transpose()
        Wfh_grad = np.dot(self.delta_f_list[t], h_prev)
        bf_grad = self.delta_f_list[t]
        Wih_grad = np.dot(self.delta_i_list[t], h_prev)
        bi_grad = self.delta_f_list[t]
        Woh_grad = np.dot(self.delta_o_list[t], h_prev)
        bo_grad = self.delta_f_list[t]
        Wch_grad = np.dot(self.delta_ct_list[t], h_prev)
        bc_grad = self.delta_ct_list[t]
        return Wfh_grad, bf_grad, Wih_grad, bi_grad,\
               Woh_grad, bo_grad, Wch_grad, bc_grad

    def update(self):
        '''
        按照梯度下降，更新权重
        '''
        self.Wfh -= self.learning_rate * self.Wfh_grad
        self.Wfx -= self.learning_rate * self.Wfx_grad
        self.bf -= self.learning_rate * self.bf_grad
        self.Wih -= self.learning_rate * self.Wih_grad
        self.Wix -= self.learning_rate * self.Wix_grad
        self.bi -= self.learning_rate * self.bi_grad
        self.Woh -= self.learning_rate * self.Woh_grad
        self.Wox -= self.learning_rate * self.Wox_grad
        self.bo -= self.learning_rate * self.bo_grad
        self.Wch -= self.learning_rate * self.Wch_grad
        self.Wcx -= self.learning_rate * self.Wcx_grad
        self.bc -= self.learning_rate * self.bc_grad


# 取数据并进行LSTM处理
df = pd.read_excel(r'workingD.xlsx', sheet_name=0)  # 取得是正调整样本

model0 = LstmLayer(1, 1, 0.01)
data0 = df.iloc[:, 0].values  # 柜位
for data00 in data0:
    model0.forward(data00)

model1 = LstmLayer(1, 1, 0.01)
data1 = df.iloc[:, 4].values  # 产消和
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


#mlp处理
rate=0.1
iteration=100
"""
初始化感知器，设置输入参数的个数，以及激活函数。
激活函数的类型
"""
x1 =T.dscalar()
# 权重向量初始化,并设置为共享变量
weights1=theano.shared(0.5)
# 偏置项初始化，并设置为共享变量
bias1=theano.shared(0.5)
#公式
result1=f(weights1 * x1 + bias1)

x2 =T.dscalar()
# 权重向量初始化,并设置为共享变量
weights2=theano.shared(0.5)
# 偏置项初始化，并设置为共享变量
bias2=theano.shared(0.5)
#公式
result2=f(weights2 * x2 + bias2)

# 权重向量初始化,并设置为共享变量
weights3=theano.shared(0.5)
# 偏置项初始化，并设置为共享变量
bias3=theano.shared(0.5)
#公式
result3=f(weights3*result1+weights3*result2+bias3)
#最终公式，输出经过LSTM和两个BP神经网络的表征值，表征值是由产消和与柜位两个元素决定的
f=theano.function([x1,x2],result3)
#取正调节样本经过LSTM前向算法处理的产消和与柜位，并把矩阵形式转化为浮点数，进行后续的BP神经网络前向算法的处理
Prlist=[]
for i in range(model0.times):
    a=float(model0.h_list[i])
    b=float(model1.h_list[i])
    Characterizationvalue0=f(a,b)
    Prlist.append(Characterizationvalue0)
print(len(Prlist))


#取负调节样本经过LSTM前向算法处理的产消和与柜位，并把矩阵形式转化为浮点数，进行后续的BP神经网络前向算法的处理
Nrlist=[]
for i in range(model2.times):
    c=float(model2.h_list[model2.times])
    d=float(model3.h_list[model3.times])
    Characterizationvalue1=f(c,d)
    Nrlist.append(Characterizationvalue1)
print(len(Nrlist))

#计算损失函数
#计算的是损失函数的分子
for i in range(model0.times):
    for j in range(model0.times):
        if i==j:
            break
        else:
            loss_molecule_value+=(np.exp(sim(Prlist[i],Prlist[j])))/temperature
print(loss_molecule_value)

#计算的是损失函数的分母
for i in range(model2.times):
    for j in range(model2.times):
        if i==j:
            break
        else:
            loss_denominator_value+=(np.exp(sim(Nrlist[i],Nrlist[j])))/temperature
print(loss_denominator_value)
#总损失函数
loss1=T.fscalar('loss1')
loss2=T.fscalar('loss2')
loss=-np.log(loss1/loss2)
dwloss=T.grad(loss,loss1)
dbloss=T.grad(loss,loss2)
fl=theano.function([loss1,loss2],[dwloss,dbloss])
dwloss0=fl([loss_molecule_value,loss_denominator_value])


print(dwloss)
# dbloss=T.grad(loss,bias)
# f= theano.function([x],[result,loss,dwloss,dbloss],updates=[(weights,weights-rate*dwloss),(bias,bias-rate*dbloss)])

