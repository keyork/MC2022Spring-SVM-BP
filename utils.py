
import numpy as np
import sympy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def X2Z_convert(X):
    
    x_1, x_2 = X
    z_1 = -3 * x_1**2 + 7 * x_1 + 1
    z_2 = -2 * x_2**2 + 2 * x_1 + 1
    Z = np.array((z_1, z_2))
    
    return Z


def get_z_data(X_list):
    
    Z_list = []

    for X in X_list:
        Z_list.append(X2Z_convert(X))
    
    return np.array(Z_list)


def sigmoid(x):
    
    return 1/(1+np.exp(-x))


def d_sigmoid(x):
    
    return np.exp(-x)/np.multiply((1+(np.exp(-x))), (1+(np.exp(-x))))


def draw_z_points(z_pos, z_neg):

    plot_x_pos = z_pos[:, 0]
    plot_y_pos = z_pos[:, 1]
    plot_x_neg = z_neg[:, 0]
    plot_y_neg = z_neg[:, 1]
    x_min = np.min([plot_x_neg, plot_x_pos])
    x_max = np.max([plot_x_neg, plot_x_pos])
    y_min = np.min([plot_y_neg, plot_y_pos])
    y_max = np.max([plot_y_neg, plot_y_pos])

    fig, ax = plt.subplots()
    ax.scatter(plot_x_pos, plot_y_pos, color='r')
    ax.scatter(plot_x_neg, plot_y_neg, color='b')
    ax.grid()
    ax.set_title(r'Sample points in $\mathcal{Z}$ space')
    ax.set_xlabel(r'$\phi_1$')
    ax.set_ylabel(r'$\phi_2$')
    plt.xlim(x_min-2, x_max+2)
    plt.ylim(y_min-2, y_max+2)
    plt.xticks(np.arange(x_min-2, x_max+3, 1))
    plt.yticks(np.arange(y_min-2, y_max+3, 1))
    ax.set_aspect(1)
    plt.show()


def draw_z_classifier(z_pos, z_neg, w1_z, b_z, w2_z):
    
    plot_x_pos = z_pos[:, 0]
    plot_y_pos = z_pos[:, 1]
    plot_x_neg = z_neg[:, 0]
    plot_y_neg = z_neg[:, 1]
    x_min = np.min([plot_x_neg, plot_x_pos])
    x_max = np.max([plot_x_neg, plot_x_pos])
    y_min = np.min([plot_y_neg, plot_y_pos])
    y_max = np.max([plot_y_neg, plot_y_pos])

    plot_x_z = np.linspace(x_min-2, x_max+2, 1000)
    plot_y_z = -(w1_z*plot_x_z + b_z) / w2_z

    fig, ax = plt.subplots()
    ax.scatter(plot_x_pos, plot_y_pos, color='r')
    ax.scatter(plot_x_neg, plot_y_neg, color='b')
    ax.plot(plot_x_z, plot_y_z, color='c')
    ax.grid()
    ax.set_title(r'Sample points and Classifier in $\mathcal{Z}$ space')
    ax.set_xlabel(r'$\phi_1$')
    ax.set_ylabel(r'$\phi_2$')
    plt.xlim(x_min-2, x_max+2)
    plt.ylim(y_min-2, y_max+2)
    plt.xticks(np.arange(x_min-2, x_max+3, 1))
    plt.yticks(np.arange(y_min-2, y_max+3, 1))
    ax.set_aspect(1)
    plt.show()


def get_classifier(data_list, label_list):

    dataset = np.array(data_list+label_list, dtype=object).reshape(2,3).T
    w1, w2, b = sympy.symbols('w1 w2 b')
    W = sympy.Matrix([[w1], [w2]])
    eq = [(W.T * data[0])[0]+b-data[1] for data in dataset]
    result = sympy.linsolve(eq, [w1, w2, b])
    return result


def draw_x_classifier(POS_DATA, NEG_DATA):

    plot_x_pos = POS_DATA[:, 0]
    plot_y_pos = POS_DATA[:, 1]
    plot_x_neg = NEG_DATA[:, 0]
    plot_y_neg = NEG_DATA[:, 1]
    x_min = np.min([plot_x_neg, plot_x_pos])
    x_max = np.max([plot_x_neg, plot_x_pos])
    y_min = np.min([plot_y_neg, plot_y_pos])
    y_max = np.max([plot_y_neg, plot_y_pos])

    plot_x_raw_1 = np.linspace(x_min-2, 0, 100000)
    plot_x_raw_2 = np.linspace(0, x_max+2, 100000)
    plot_x_x_1 = plot_x_raw_1[np.where(3*plot_x_raw_1*plot_x_raw_1-3*plot_x_raw_1-1 >= 0)]
    plot_x_x_2 = plot_x_raw_2[np.where(3*plot_x_raw_2*plot_x_raw_2-3*plot_x_raw_2-1 >= 0)]
    plot_x_y_1_1 = np.sqrt(3*plot_x_x_1*plot_x_x_1-3*plot_x_x_1-1)/2
    plot_x_y_2_1 = -np.sqrt(3*plot_x_x_1*plot_x_x_1-3*plot_x_x_1-1)/2
    plot_x_y_1_2 = np.sqrt(3*plot_x_x_2*plot_x_x_2-3*plot_x_x_2-1)/2
    plot_x_y_2_2 = -np.sqrt(3*plot_x_x_2*plot_x_x_2-3*plot_x_x_2-1)/2
    
    fig, ax = plt.subplots()
    ax.scatter(plot_x_pos, plot_y_pos, color='r')
    ax.scatter(plot_x_neg, plot_y_neg, color='b')
    ax.plot(plot_x_x_1, plot_x_y_1_1, color='c')
    ax.plot(plot_x_x_1, plot_x_y_2_1, color='c')
    ax.plot(plot_x_x_2, plot_x_y_1_2, color='c')
    ax.plot(plot_x_x_2, plot_x_y_2_2, color='c')
    ax.grid()
    ax.set_title(r'Sample points and Classifier in $\mathcal{X}$ space')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    plt.xlim(x_min-2, x_max+2)
    plt.ylim(y_min-2, y_max+2)
    plt.xticks(np.arange(x_min-2, x_max+3, 1))
    plt.yticks(np.arange(y_min-2, y_max+3, 1))
    ax.set_aspect(1)
    plt.show()


def train_mlp(mlp):

    # print('更新前的参数：')
    # print('w1:\n{}'.format(mlp.w1))
    # print('w2:\n{}'.format(mlp.w2))
    train_data = np.matrix((0.6, 0.1)).T
    train_label = np.matrix((1.0, 0.0)).T
    learning_rate = 0.1
    mlp.forward(train_data)
    mlp.get_loss(train_label)
    mlp.back_propagation(train_data, train_label)
    mlp.update_w(learning_rate)
    # print('更新后的参数：')
    # print('w1:\n{}'.format(mlp.w1))
    # print('w2:\n{}'.format(mlp.w2))


def train_mlp_torch(model):

    train_data =  torch.tensor([0.6, 0.1], requires_grad=True)
    train_label = torch.tensor([1.0, 0.0], requires_grad=True)
    model.train()
    # print('更新前的参数：')
    # print(model.fc1.weight)
    # print(model.fc2.weight)
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    y_pred = model.forward(train_data)
    loss = loss_fn(y_pred, train_label)
    loss.backward()
    optimizer.step()
    # print('更新后的参数：')
    # print(model.fc1.weight)
    # print(model.fc2.weight)