import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import classes
from torchvision.transforms import transforms


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(5, translate=(2/32, 2/32), scale=(0.95, 1.05), resample=PIL.Image.NEAREST),
        transforms.ToTensor(),
        transforms.RandomChoice([
            transforms.Normalize((0.5, 0.5, 0.5), (0.9, 0.9, 0.9)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.7, 0.7, 0.7)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
