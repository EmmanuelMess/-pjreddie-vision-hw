import PIL
import torch
import torchvision
import torchvision.transforms as transforms


class CifarLoader(object):
	"""docstring for CifarLoader"""
	def __init__(self, args):
		super(CifarLoader, self).__init__()
		transform = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomAffine(5, translate=(2 / 32, 2 / 32), scale=(0.95, 1.05), resample=PIL.Image.NEAREST),
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
		self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True, num_workers=2)

		testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
		self.testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, num_workers=2)

		self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

