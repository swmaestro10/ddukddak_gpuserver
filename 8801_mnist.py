from flask import Flask, request

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from IPython.display import Image
from IPython import display
import matplotlib.pyplot as plt


def code_mnist(training_num, test_num, learning_rate, num_epochs, _activation):
    batch_size = 10
    trn_dataset = datasets.MNIST('./data/', download=True, train=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = datasets.MNIST("./data/", download=False, train=False, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    num_batches = len(trn_loader)
    data = next(iter(trn_loader))
    img, label = data

    row_num = 1
    col_num = 4
    fig, ax = plt.subplots(row_num, col_num, figsize=(6, 6))
    for i, j in itertools.product(range(row_num), range(col_num)):
        ax[j].get_xaxis().set_visible(False)
        ax[j].get_yaxis().set_visible(False)

    for k in range(row_num * col_num):
        i = k // col_num
        j = k % col_num
        ax[j].cla()
        ax[j].imshow(img[k, :].data.cpu().numpy().reshape(28, 28), cmap='Greys')

    use_cuda = torch.cuda.is_available()

    class CNNClassifier(nn.Module):

        def __init__(self):
            super(CNNClassifier, self).__init__()
            conv1 = nn.Conv2d(1, 6, 5, 1)
            pool1 = nn.MaxPool2d(2)
            conv2 = nn.Conv2d(6, 16, 5, 1)
            pool2 = nn.MaxPool2d(2)

            if _activation[0] == 'R' and _activation[1] == 'R':
                self.conv_module = nn.Sequential(conv1, nn.ReLU(), pool1, conv2, nn.ReLU(), pool2)
            elif _activation[0] == 'R' and _activation[1] == 'L':
                self.conv_module = nn.Sequential(conv1, nn.ReLU(), pool1, conv2, nn.LogSigmoid(), pool2)
            elif _activation[0] == 'L' and _activation[1] == 'R':
                self.conv_module = nn.Sequential(conv1, nn.LogSigmoid(), pool1, conv2, nn.ReLU(), pool2)
            elif _activation[0] == 'L' and _activation[1] == 'L':
                self.conv_module = nn.Sequential(conv1, nn.LogSigmoid(), pool1, conv2, nn.LogSigmoid(), pool2)

            fc1 = nn.Linear(16 * 4 * 4, 120)
            fc2 = nn.Linear(120, 84)
            fc3 = nn.Linear(84, 10)

            if _activation[2] == 'R' and _activation[3] == 'R':
                self.fc_module = nn.Sequential(fc1, nn.ReLU(), fc2, nn.ReLU(), fc3)
            elif _activation[2] == 'R' and _activation[3] == 'L':
                self.fc_module = nn.Sequential(fc1, nn.ReLU(), fc2, nn.LogSigmoid(), fc3)
            elif _activation[2] == 'L' and _activation[3] == 'R':
                self.fc_module = nn.Sequential(fc1, nn.LogSigmoid(), fc2, nn.ReLU(), fc3)
            elif _activation[2] == 'L' and _activation[3] == 'L':
                self.fc_module = nn.Sequential(fc1, nn.LogSigmoid(), fc2, nn.LogSigmoid(), fc3)

            if use_cuda:
                self.conv_module = self.conv_module.cuda()
                self.fc_module = self.fc_module.cuda()

        def forward(self, x):
            out = self.conv_module(x)
            dim = 1
            for d in out.size()[1:]:
                dim = dim * d
            out = out.view(-1, dim)
            out = self.fc_module(out)
            return F.softmax(out, dim=1)

    cnn = CNNClassifier()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    row_num = 1
    col_num = 4
    fig, ax = plt.subplots(row_num, col_num, figsize=(6, 6))
    for i, j in itertools.product(range(row_num), range(col_num)):
        ax[j].get_xaxis().set_visible(False)
        ax[j].get_yaxis().set_visible(False)

    trn_loss_list = []
    val_loss_list = []
    check = 0
    for epoch in range(num_epochs):
        trn_loss = 0.0
        training_cnt = 0

        for i, data in enumerate(trn_loader):
            check += 1
            training_cnt += batch_size
            x, label = data
            if use_cuda:
                x = x.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            model_output = cnn(x)
            loss = criterion(model_output, label)
            loss.backward()
            optimizer.step()

            trn_loss += loss.item()
            del loss
            del model_output

            if (i + 1) % 100 == 0:
                with torch.no_grad():
                    val_loss = 0.0
                    corr_num = 0
                    total_num = 0
                    test_cnt = 0
                    for j, val in enumerate(val_loader):
                        test_cnt += batch_size
                        val_x, val_label = val
                        if use_cuda:
                            val_x = val_x.cuda()
                            val_label = val_label.cuda()
                        val_output = cnn(val_x)
                        v_loss = criterion(val_output, val_label)
                        val_loss += v_loss
                        model_label = val_output.argmax(dim=1)
                        corr = val_label[val_label == model_label].size(0)
                        corr_num += corr
                        total_num += val_label.size(0)
                        if (test_cnt >= test_num): break

                for k in range(row_num * col_num):
                    ii = k // col_num
                    jj = k % col_num
                    ax[jj].cla()
                    ax[jj].imshow(val_x[k, :].data.cpu().numpy().reshape(28, 28), cmap='Greys')

                display.clear_output(wait=True)
                display.display(plt.gcf())
                plt.show()
                print("acc: {:.2f}".format(corr_num / total_num * 100))
                print("label: {}".format(val_label[:row_num * col_num]))
                print("prediction: {}".format(val_output.argmax(dim=1)[:row_num * col_num]))
                del val_output
                del v_loss

                print("epoch: {}/{} | step: {}/{} | trn loss: {:.4f} | val loss: {:.4f}".format(
                    epoch + 1, num_epochs, i + 1, num_batches, trn_loss / 100, val_loss / len(val_loader)
                ))

                trn_loss_list.append(trn_loss / 100)
                val_loss_list.append(val_loss / len(val_loader))
                trn_loss = 0.0

                if (training_cnt >= training_num): break

    with torch.no_grad():
        corr_num = 0
        total_num = 0
        for j, val in enumerate(val_loader):
            val_x, val_label = val
            if use_cuda:
                val_x = val_x.cuda()
                val_label = val_label.cuda()

            val_output = cnn(val_x)
            model_label = val_output.argmax(dim=1)
            corr = val_label[val_label == model_label].size(0)
            corr_num += corr
            total_num += val_label.size(0)

    print("acc: {:.2f}".format(corr_num / total_num * 100))

    return str(corr_num / total_num * 100)


app = Flask(__name__)


@app.route('/')
def index():
    return 'DDUKDDAK-Gpu-Server MNIST running on port 8801!'


@app.route('/run', methods=['POST'])
def run():
    code = request.form['code']

    params = code.split(';')

    _train = None
    _test = None
    _learn = None
    _epoch = None
    _activation = []

    for i in params:
        if i.find('training_num') != -1:
            ii = i.find('=')
            num = i[ii+2:-1]
            _train = int(num)
        elif i.find('test_num') != -1:
            ii = i.find('=')
            num = i[ii+2:-1]
            _test = int(num)
        elif i.find('learning_rate') != -1:
            ii = i.find('=')
            num = i[ii+2:-1]
            _learn = float(num)
        elif i.find('num_epochs') != -1:
            ii = i.find('=')
            num = i[ii+2:-1]
            _epoch = int(num)
        elif i.find('LogSigmoid') != -1:
            _activation.append('L')
        elif i.find('ReLU') != -1:
            _activation.append('R')

    print('MNIST params : ' + str(_train) + ' ' + str(_test) + ' ' + str(_learn) + ' ' + str(_epoch) + ' ' + str(_activation))

    return code_mnist(_train, _test, _learn, _epoch, _activation)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8801)
