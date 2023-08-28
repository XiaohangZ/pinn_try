import torch
import torch.nn as nn
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from torch import autograd
from data_loader import *
from config import *
from metrics import *
from utils import *

class Net(nn.Module):
    def __init__(self, inputNode=2, hiddenNode=256, outputNode=1):
        super(Net, self).__init__()
        # Define Hyperparameters
        self.inputLayerSize = inputNode
        self.outputLayerSize = outputNode
        self.hiddenLayerSize = hiddenNode

        # weights
        self.Linear1 = nn.Linear(self.inputLayerSize, self.hiddenLayerSize)
        self.Linear2 = nn.Linear(self.hiddenLayerSize, self.outputLayerSize)
        self.activation = torch.nn.Sigmoid()

    def forward(self, X):
        out1 = self.Linear1(X)
        out2 = self.activation(out1)
        out3 = self.Linear2(out2)
        return out3

net = Net(inputNode=2, hiddenNode=256, outputNode=1)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
num_epochs = 100





def pinn_loss(r_true, r_pred, delta, U, r_prv):
    T_dotr = 221/5
    T_U_dotr = -62/45
    T_U2_dotr = 449/180
    T_U3_dotr = -193/1620
    K_delta = -7/100
    K_U_delta = 1/360
    K_U2_delta = 1/180
    K_U3_delta = -1/3024
    N_r = 1
    N_r3 = 1/2
    N_U_r3 = -43/180
    N_U2_r3 = 1/18
    N_U3_r3 = -1/324
    sampling_time = 1
    assert r_true.shape == r_pred.shape == delta.shape == U.shape
    F_rudder = K_delta * delta + K_U_delta * U * delta + K_U2_delta * U*U * delta + K_U3_delta * U*U*U * delta
    F_hydro = N_r * r_pred + N_r3 * r_pred*r_pred*r_pred + N_U_r3 * U * r_pred*r_pred*r_pred + N_U2_r3 * U*U * r_pred*r_pred*r_pred + N_U3_r3 * U*U*U * r_pred*r_pred*r_pred
    r_dot = (r_pred - r_prv)/sampling_time
    R = F_rudder - F_hydro - (T_dotr + T_U_dotr * U + T_U2_dotr * U*U + T_U3_dotr * U*U*U) * r_dot
    return torch.mean(R*R)

# def data_loss():
#     pass



def train(model=None, SavingName=None, train_loader=None, val_loader=None, optimizer=None, num_epochs = num_epochs):
    total_step = len(train_loader)

    dataset = r'D:\OneDrive\Captures\Masterarbeit\code_of_masterarbeit\code_of_pinn_try\train.csv'
    df = pd.read_csv(dataset)
    U = df['U'].values
    U = torch.from_numpy(U)
    # U = torch.unsqueeze(U,dim = 1)
    delta = df['delta'].values
    delta = torch.from_numpy(delta)
    # delta = torch.unsqueeze(delta,dim = 1)
    r_true = df['r'].values
    r_true = torch.from_numpy(r_true)



    for epoch in range(num_epochs):
        for i, (signals, labels) in enumerate(train_loader):

            # Forward pass
            outputs = model(signals)
            r_pred = model(signals)
            r_prv = model(signals)

            loss_function = nn.MSELoss()
            MSE_r = loss_function(labels.to(torch.float32), outputs.to(torch.float32))


            # MSE_r = data_loss()
            MSE_R = pinn_loss(r_true, r_pred, delta, U, r_prv)

            loss = MSE_r + MSE_R

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

            if epoch % 10 == 0:
                with torch.no_grad():
                    model.eval()

                    pred, gt = [], []

                    for signalsV, labelsV in val_loader:
                        labelsV = labelsV.to(device)
                        signalsV = signalsV.to(device)

                        outputsV = model(signalsV)

                        gt.extend(labelsV.cpu().numpy()[0])
                        pred.extend(outputsV[0].round().cpu().numpy())

                    gt = np.asarray(gt, np.float32)
                    pred = np.asarray(pred)

                    print('Val Accuracy of the model on the {} epoch: {} %'.format(epoch, accuracy(pred, gt)))

                model.train()

    # Save the model checkpoint
    checkDirMake(os.path.dirname(SavingName))
    torch.save(model.state_dict(), SavingName)


def test(model=None, SavingName=None, test_loader=None):
    model.load_state_dict(torch.load(SavingName))
    # Test the model

    model.eval()
    with torch.no_grad():
        pred, gt = [], []

        for signals, labels in test_loader:
            signals = signals.to(device)
            outputs = model(signals)
            outputs = outputs.round().cpu().numpy()

            gt.extend(labels.cpu().numpy()[0])
            pred.extend(outputs[0])

        gt = np.asarray(gt, np.float32)
        pred = np.asarray(pred)

        print('Test Accuracy of the model test samples: {} %'.format(accuracy(pred, gt)))

train(model = net, SavingName='./checkpoints/nn.ckpt', train_loader = train_loader, val_loader=val_loader,num_epochs = num_epochs)
test(model = net, SavingName='./checkpoints/nn.ckpt', test_loader=test_loader)




