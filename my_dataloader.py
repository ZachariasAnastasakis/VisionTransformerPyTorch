import torch
from torchvision import datasets, transforms


class LoadDataset:
    """
    Class to load dataset.

    Arguments
    ---------
    dataset: str, name of the dataset.
    batch_size:, int, batch size
    mode: str, whether we train or test our model

    Returns
    -------
    train_data_loader: torch.utils.data.DataLoader(), the train or test dataloader

    """

    def __init__(self, dataset="MNIST", batch_size=16, mode="train"):
        if mode == "train":
            self.mode = True
        else:
            self.mode = False
        self.batch_size = batch_size
        if dataset == "MNIST":
            transform = transforms.ToTensor()
            self.data = datasets.MNIST(root='./data', train=self.mode, download=True, transform=transform)
        elif dataset == "CIFAR10":
            if mode == "train":
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

            self.data = datasets.CIFAR10(root='./data', train=self.mode, download=True, transform=transform)
        else:
            raise ValueError("Please select one of the following datasets: 'CIFAR10' or 'MNIST'.")

    def DataLoader(self):
        train_data_loader = torch.utils.data.DataLoader(dataset=self.data, batch_size=self.batch_size,
                                                        shuffle=self.mode)

        return train_data_loader
