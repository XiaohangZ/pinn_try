import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import autograd

class Net(nn.Module):
    def __init__(self, NL, NN): # NL n个l（线性，全连接）隐藏层， NN 输入数据的维数，
        # NL是有多少层隐藏层
        # NN是每层的神经元数量
        super(Net, self).__init__()
        self.input_layer = nn.Linear(1, NN)
        self.hidden_layer = nn.Linear(NN,int(NN/2))
        self.output_layer = nn.Linear(int(NN/2), 1)

    def forward(self, x):
        out = torch.tanh(self.input_layer(x))
        out = torch.tanh(self.hidden_layer(out))
        out_final = self.output_layer(out)
        return out_final


net = Net(4,20)
mse_cost_function = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(),lr=1e-4)
def ode_01(x,net):
    y=net(x)
    y_x = autograd.grad(y, x,grad_outputs=torch.ones_like(net(x)),create_graph=True)[0]
    return y-y_x   # y-y' = 0

# requires_grad=True).unsqueeze(-1)

plt.ion()  # 动态图
iterations=200000
for epoch in range(iterations):

    optimizer.zero_grad()  # 梯度归0

    ## 求边界条件的损失函数
    x_0 = torch.zeros(2000, 1)
    y_0 = net(x_0)
    mse_i = mse_cost_function(y_0, torch.ones(2000, 1))  # f(0) - 1 = 0

    ## 方程的损失函数
    x_in = np.random.uniform(low=0.0, high=2.0, size=(2000, 1))
    pt_x_in = autograd.Variable(torch.from_numpy(x_in).float(), requires_grad=True)  # x 随机数
    pt_y_colection=ode_01(pt_x_in,net)
    pt_all_zeros= autograd.Variable(torch.from_numpy(np.zeros((2000,1))).float(), requires_grad=False)
    mse_f=mse_cost_function(pt_y_colection, pt_all_zeros)  # y-y' = 0

    loss = mse_i + mse_f
    loss.backward()  # 反向传播
    optimizer.step()  # 优化下一步。This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

    if epoch%1000==0:
            y = torch.exp(pt_x_in)  # y 真实值
            y_train0 = net(pt_x_in) # y 预测值
            print(epoch, "Traning Loss:", loss.data)
            print(f'times {epoch}  -  loss: {loss.item()} - y_0: {y_0}')
            plt.cla()
            plt.scatter(pt_x_in.detach().numpy(), y.detach().numpy())
            plt.scatter(pt_x_in.detach().numpy(), y_train0.detach().numpy(),c='red')
            plt.pause(0.1)
