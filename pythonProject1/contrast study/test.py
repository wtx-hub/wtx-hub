import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops

# ops.reset_default_graph()
# sess=tf.Session()
verification_length=1
input_size=1

df=pd.read_excel(r'workingC.xlsx')
data=df.iloc[:,0].values

# normalized_train_data=(data-np.mean(data,axis=0))/np.std(data,axis=0)  #标准化
# normalized_test_data=(test_data-np.mean(test_data,axis=0))/np.std(test_data,axis=0)  #标准化
# s=np.mean(data,axis=0)
# m=np.std(data,axis=0)
# s1=np.mean(test_data,axis=0)
# m1=np.std(test_data,axis=0)
# normalized_train_data = normalized_train_data.tolist()
# normalized_test_data = normalized_test_data.tolist()
train_x=[]
# train_y=[]
for i in range(len(data)):
    x = data[i:i + verification_length]
    train_x.append(x)
train_x = np.array(train_x)
# train_y = np.array(train_y)
train_x=train_x.reshape(-1,1)
# train_x = np.array(train_x)
# train_y = np.array(train_y)
# train_y=train_y.reshape(-1,1)
# print(train_x.shape)
# print(train_y.shape)
#
# test_x=[]
# test_y=[]
# for i in range(len(normalized_test_data) - verification_length):
#     x = normalized_test_data[i:i + verification_length]
#     y = normalized_test_data[i + verification_length]
#     test_x.append(x)
#     test_y.append(y)
#
# test_x = np.array(test_x)
# test_y = np.array(test_y)
# test_y=test_y.reshape(-1,1)
# print(test_x.shape)
# print(test_y.shape)
#
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

model=LstmLayer(1,90,0.001)
model.forward(train_x)
print(model.h_list[model.times])
# for i in range(0,50):
#     train_x1=np.array(train_x[:,i])
#     train_x2=train_x1.reshape(-1,1)
#     model.forward(train_x2)
# h=model.h_list
# h=np.array(h)
# pred=h[50,:,:]
# print(pred.shape)
# pred = np.reshape(pred, (pred.size,))
# print(pred.shape)
# def normalized(x,y):
#     normalized_x=x*np.std(y,axis=0)+np.mean(y,axis=0)
#     return normalized_x
# pred=normalized(pred,train_data)
# print(pred)

#
# def loss(x, y):
#     loss_data=np.multiply(np.subtract(x,y),np.subtract(x,y))
#     return loss_data
# print(type(train_y))
# print(type(pred))
# print(train_y.shape)
# print(pred.shape)
# train_y=np.reshape(train_y, (train_y.size,))
# print(train_y.shape)
# train_y1=train_y*np.std(train_data,axis=0)+np.mean(train_data,axis=0)
# loss1=loss(pred,train_y1)
# print(loss1.shape)
# print((loss1))

# def backward(self, x, delta_h, activator):
#     '''
#     实现LSTM训练算法
#     '''
#     self.calc_delta(delta_h, activator)
#     self.calc_gradient(x)
#
#     def calc_delta(self, delta_h, activator):
#         # 初始化各个时刻的误差项
#         self.delta_h_list = self.init_delta()  # 输出误差项
#         self.delta_o_list = self.init_delta()  # 输出门误差项
#         self.delta_i_list = self.init_delta()  # 输入门误差项
#         self.delta_f_list = self.init_delta()  # 遗忘门误差项
#         self.delta_ct_list = self.init_delta()  # 即时输出误差项
#         # 保存从上一层传递下来的当前时刻的误差项
#         self.delta_h_list[-1] = delta_h
#         # 迭代计算每个时刻的误差项
#         for k in range(self.times, 0, -1):
#             self.calc_delta_k(k)
#
#     def init_delta(self):
#         '''
#         初始化误差项
#         '''
#         delta_list = []
#         for i in range(self.times + 1):
#             delta_list.append(np.zeros(
#                 (self.state_width, 1)))
#         return delta_list
#
#     def calc_delta_k(self, k):
#         '''
#         根据k时刻的delta_h，计算k时刻的delta_f、
#         delta_i、delta_o、delta_ct，以及k-1时刻的delta_h
#         '''
#         # 获得k时刻前向计算的值
#         ig = self.i_list[k]
#         og = self.o_list[k]
#         fg = self.f_list[k]
#         ct = self.ct_list[k]
#         c = self.c_list[k]
#         c_prev = self.c_list[k - 1]
#         tanh_c = self.output_activator.forward(c)
#         delta_k = self.delta_h_list[k]
#         # 根据式9计算delta_o
#         delta_o = (delta_k * tanh_c *
#                    self.gate_activator.backward(og))
#         delta_f = (delta_k * og *
#                    (1 - tanh_c * tanh_c) * c_prev *
#                    self.gate_activator.backward(fg))
#         delta_i = (delta_k * og *
#                    (1 - tanh_c * tanh_c) * ct *
#                    self.gate_activator.backward(ig))
#         delta_ct = (delta_k * og *
#                     (1 - tanh_c * tanh_c) * ig *
#                     self.output_activator.backward(ct))
#         delta_h_prev = (
#                 np.dot(delta_o.transpose(), self.Woh) +
#                 np.dot(delta_i.transpose(), self.Wih) +
#                 np.dot(delta_f.transpose(), self.Wfh) +
#                 np.dot(delta_ct.transpose(), self.Wch)
#         ).transpose()
#         # 保存全部delta值
#         self.delta_h_list[k - 1] = delta_h_prev
#         self.delta_f_list[k] = delta_f
#         self.delta_i_list[k] = delta_i
#         self.delta_o_list[k] = delta_o
#         self.delta_ct_list[k] = delta_ct
#
#         def calc_gradient(self, x):
#             # 初始化遗忘门权重梯度矩阵和偏置项
#             self.Wf_grad, self.bf_grad = (
#                 self.init_weight_gradient_mat())
#             # 初始化输入门权重梯度矩阵和偏置项
#             self.Wi_grad, self.bi_grad = (
#                 self.init_weight_gradient_mat())
#             # 初始化输出门权重梯度矩阵和偏置项
#             self.Wo_grad, self.bo_grad = (
#                 self.init_weight_gradient_mat())
#             # 初始化单元状态权重梯度矩阵和偏置项
#             self.Wc_grad, self.bc_grad = (
#                 self.init_weight_gradient_mat())
#             # 计算对上一次输出h的权重梯度
#             for t in range(self.times, 0, -1):
#                 # 计算各个时刻的梯度
#                 (Wf_grad, bf_grad,
#                  Wi_grad, bi_grad,
#                  Wo_grad, bo_grad,
#                  Wc_grad, bc_grad) = (
#                     self.calc_gradient_t(t))
#                 # 实际梯度是各时刻梯度之和
#                 self.Wf_grad += Wf_grad
#                 self.bf_grad += bf_grad
#                 self.Wi_grad += Wi_grad
#                 self.bi_grad += bi_grad
#                 self.Wo_grad += Wo_grad
#                 self.bo_grad += bo_grad
#                 self.Wc_grad += Wc_grad
#                 self.bc_grad += bc_grad
#                 # print('-----%d-----',%t)
#                 print(Wf_grad)
#                 print(self.Wf_grad)
#             # 计算对本次输入x的权重梯度
#             xt = x.transpose()
#             self.Wf_grad = np.dot(self.delta_f_list[-1], xt)
#             self.Wi_grad = np.dot(self.delta_i_list[-1], xt)
#             self.Wo_grad = np.dot(self.delta_o_list[-1], xt)
#             self.Wc_grad = np.dot(self.delta_ct_list[-1], xt)
#
#         def init_weight_gradient_mat(self):
#             '''
#             初始化权重矩阵
#             '''
#             Wf_grad = np.zeros((self.state_width,
#                                 self.state_width))
#             b_grad = np.zeros((self.state_width, 1))
#             return Wf_grad, b_grad
#
#         def calc_gradient_t(self, t):
#             '''
#             计算每个时刻t权重的梯度
#             '''
#             h_prev = self.h_list[t - 1].transpose()
#             Wf_grad = np.dot(self.delta_f_list[t], h_prev)
#             bf_grad = self.delta_f_list[t]
#             Wi_grad = np.dot(self.delta_i_list[t], h_prev)
#             bi_grad = self.delta_f_list[t]
#             Wo_grad = np.dot(self.delta_o_list[t], h_prev)
#             bo_grad = self.delta_f_list[t]
#             Wc_grad = np.dot(self.delta_ct_list[t], h_prev)
#             bc_grad = self.delta_ct_list[t]
#             return Wf_grad, bf_grad, Wi_grad, bi_grad, \
#                    Wo_grad, bo_grad, Wc_grad, bc_grad
#
#         def update(self):
#             '''
#             按照梯度下降，更新权重
#             '''
#             self.Wf -= self.learning_rate * self.Wf_grad
#             # self.Wfx -= self.learning_rate * self.Whx_grad
#             self.bf -= self.learning_rate * self.bf_grad
#             self.Wi -= self.learning_rate * self.Wi_grad
#             # self.Wix -= self.learning_rate * self.Whi_grad
#             self.bi -= self.learning_rate * self.bi_grad
#             self.Wo -= self.learning_rate * self.Wo_grad
#             # self.Wox -= self.learning_rate * self.Wox_grad
#             self.bo -= self.learning_rate * self.bo_grad
#             self.Wc -= self.learning_rate * self.Wc_grad
#             # self.Wcx -= self.learning_rate * self.Wcx_grad
#             self.bc -= self.learning_rate * self.bc_grad
#
#         def reset_state(self):
#             # 当前时刻初始化为t0
#             self.times = 0
#             # 各个时刻的单元状态向量c
#             self.c_list = self.init_state_vec()
#             # 各个时刻的输出向量h
#             self.h_list = self.init_state_vec()
#             # 各个时刻的遗忘门f
#             self.f_list = self.init_state_vec()
#             # 各个时刻的输入门i
#             self.i_list = self.init_state_vec()
#             # 各个时刻的输出门o
#             self.o_list = self.init_state_vec()
#             # 各个时刻的即时状态c~
#             self.ct_list = self.init_state_vec()
#
#         def data_set():
#             x = [np.array([[1], [2], [3]]),
#                  np.array([[2], [3], [4]])]
#             d = np.array([[1], [2]])
#             return x, d
#
# def gradient_check():
#     '''
#     梯度检查
#     '''
#      # 设计一个误差函数，取所有节点输出项之和
#     error_function = lambda o: o.sum()
#     lstm = LstmLayer(3, 2, 1e-3)
#      # 计算forward值
#     x, d = data_set()
#     lstm.forward(x[0])
#     lstm.forward(x[1])
#     # 求取sensitivity map
#     sensitivity_array = np.ones(lstm.h_list[-1].shape,dtype=np.float64)
#     # 计算梯度
#     lstm.backward(x[1], sensitivity_array, IdentityActivator())
#     # 检查梯度
#     epsilon = 10e-4
#     for i in range(lstm.Wf.shape[0]):
#         for j in range(lstm.Wf.shape[1]):
#             lstm.Wf[i, j] += epsilon
#             lstm.reset_state()
#             lstm.forward(x[0])
#             lstm.forward(x[1])
#             err1 = error_function(lstm.h_list[-1])
#             lstm.Wf[i, j] -= 2 * epsilon
#             lstm.reset_state()
#             lstm.forward(x[0])
#             lstm.forward(x[1])
#             err2 = error_function(lstm.h_list[-1])
#             expect_grad = (err1 - err2) / (2 * epsilon)
#             lstm.Wf[i, j] += epsilon
#             print('weights(%d,%d): expected - actural %.4e - %.4e'% (i, j, expect_grad, lstm.Wf_grad[i, j]))
#     return lstm