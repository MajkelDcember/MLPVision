import torch
device = torch.device("cpu")
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np



transform = transforms.ToTensor()
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

batch_size = 50
num_workers = 0
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
import matplotlib.pyplot as plt
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()
plt.imshow(images[0].reshape((28, 28)), cmap="gray")
#plt.show()
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

model = Net()
#print(model)
criterion = nn.CrossEntropyLoss().cpu()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
n_epochs=30
model.train().cpu()

for n in range(n_epochs):
    train_loss=0.0
    for x,y in train_loader:
        optimizer.zero_grad()
        y_eval=model(x).cpu()
        loss=criterion(y_eval,y)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()*x.size(0)
    train_loss = train_loss / len(train_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        n + 1,
        train_loss
    ))



test_loss = 0.0
class_correct = list(0 for i in range(10))
class_total = list(0 for i in range(10))

model.eval().cpu()

for data, target in test_loader:

    output = model(data)

    loss = criterion(output, target)

    test_loss += loss.item()*data.size(0)

    _, pred = torch.max(output, 1)

    correct = (pred == target.view_as(pred)).squeeze()

    for i in range(batch_size):
        label = target[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1


test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)')

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))