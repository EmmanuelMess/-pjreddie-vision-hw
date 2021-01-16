import os

import torch
import wandb

from dataloader import CifarLoader
from utils import argParser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(net, dataloader, optimizer, criterion, epoch):
    net.to(device)

    running_loss = 0.0
    total_loss = 0.0

    #summary(net, (3, 32, 32))

    for i, data in enumerate(dataloader.trainloader, 0):
        # get the inputs
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        total_loss += loss.item()
        if (i + 1) % 2000 == 0:    # print every 2000 mini-batches
            net.log('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    net.log('Final Summary:   loss: %.3f' %
          (total_loss / i))

    wandb.log({"loss": total_loss / i})


def test(net, dataloader, tag=''):
    correct = 0
    total = 0
    if tag == 'Train':
        dataTestLoader = dataloader.trainloader
    else:
        dataTestLoader = dataloader.testloader
    with torch.no_grad():
        for data in dataTestLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    net.log('%s Accuracy of the network: %d %%' % (tag,
        100 * correct / total))

    wandb.log({'{} Accuracy of the network'.format("tag"): correct / total})
"""
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in dataTestLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        net.log('%s Accuracy of %5s : %2d %%' % (
            tag, dataloader.classes[i], 100 * class_correct[i] / class_total[i]))

        wandb.log({'{} Accuracy of {}'.format("tag", dataloader.classes[i]): class_correct[i] / class_total[i]})
"""
def main():

    args = argParser()

    cifarLoader = CifarLoader(args)
    net = args.model()
    print('The log is recorded in ')
    print(net.logFile.name)

    if os.path.isfile('model.pth'):
        net.load_state_dict(torch.load('model.pth'))

    criterion = net.criterion()
    optimizer = net.optimizer()

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        net.adjust_learning_rate(optimizer, epoch, args)
        train(net, cifarLoader, optimizer, criterion, epoch)
        if epoch % 10 == 0: # Comment out this part if you want a faster training
            test(net, cifarLoader, 'Train')
            test(net, cifarLoader, 'Test')

    torch.save(net.state_dict(), 'model.pth')
    print('The log is recorded in ')
    print(net.logFile.name)

if __name__ == '__main__':
    wandb.init(project='test-cifar10')
    main()
