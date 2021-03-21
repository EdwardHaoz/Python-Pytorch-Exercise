# 2021-03-21 Pytoch Test: MNIST
# By: Zhou Hao

import torch
import torch.optim
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10
INTERVAL = 10

# download data
train_set = datasets.MNIST(root="D:/code", train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((1.307,), (0.3081,))
]))

test_set = datasets.MNIST(root="D:/code", train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((1.307,), (0.3081,))
]))

# load data
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)

# show example
"""
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

for i in range(6):
  plt.subplot(2, 3, i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
plt.figure()
plt.show()
"""


# build network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 1->Gray 10->Out
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(160, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        input_size = x.size(0)  # batch_size
        x = F.relu(F.max_pool2d(self.conv1(x), 2, 2))
        # in_size:batch_size*1*28*28 conv1_out:batch_size*10*(28-5+1)*24 pool_out:batch_size*10*12*12
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        # in_size:batch_size*10*12*12 conv2_out:batch_size*10*(12-5+1)*8 pool_out:batch_size*10*4*4
        x = self.dropout(x)
        x = x.view(input_size, -1)
        x = self.fc1(x)
        # in_size:batch_size*10*4*4 fc1_out:batch_size*500
        x = self.dropout(x)
        x = self.fc2(x)
        # in_size:batch_size*500 fc2_out:batch_size*10
        return F.log_softmax(x, dim=1)


# define train
model = Net().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())
train_loss = []
train_count = []


def train_model(epoch):
    model.train()
    for batch_idx, (img_data, img_label) in enumerate(train_loader):
        img_data, img_label = img_data.to(DEVICE), img_label.to(DEVICE)
        optimizer.zero_grad()
        output = model(img_data)
        loss = F.nll_loss(output, img_label)
        pred = output.argmax(dim=1)
        loss.backward()
        optimizer.step()
        if batch_idx % INTERVAL == 0:
            print('Train Epoch:{} [{}/{} ({:.6f}%)] Loss:{:.6f}\t '.format(
                epoch, batch_idx * len(img_data), len(train_loader.dataset),
                100 * batch_idx * len(img_data) / len(train_loader.dataset), loss.item()
            ))
        train_loss.append(loss.item())
        train_count.append((epoch - 1) * len(train_loader.dataset) + batch_idx * BATCH_SIZE_TRAIN)
        torch.save(model.state_dict(), 'D:/code/MNIST/result/model.pth')
        torch.save(optimizer.state_dict(), 'D:/code/MNIST/result/optimizer.pth')


# define test
test_loss = []
test_count = []


def test_model(epoch):
    model.eval()
    correct = 0.0
    loss = 0.0
    with torch.no_grad():
        for img_data, img_label in test_loader:
            img_data, img_label = img_data.to(DEVICE), img_label.to(DEVICE)
            output = model(img_data)
            pred = output.argmax(dim=1)
            correct += pred.eq(img_label.data.view_as(pred)).sum()
            loss += F.nll_loss(output, img_label).item()
        loss /= len(test_loader.dataset)
        correct /= len(test_loader.dataset)
        test_loss.append(loss)
        test_count.append(epoch*len(train_loader.dataset))
        print('\nTest Result {} -- Correct:({:.6f}%) Loss:{:.6f}\n '.format(len(test_loader.dataset), 100 * correct, loss))


test_model(1)
for i in range(1, EPOCHS + 1):
    train_model(i)
    test_model(i)

plt.plot(train_count, train_loss, color='blue')
plt.scatter(test_count, test_loss, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.figure()
plt.show()
