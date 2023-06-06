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

    def __init__(self, dataset="MNIST", batch_size=32, mode="train"):
        if mode == "train":
            self.mode = True
        else:
            self.mode = False
        self.batch_size = batch_size
        transform = transforms.ToTensor()
        if dataset == "MNIST":
            self.data = datasets.MNIST(root='./data', train=self.mode, download=True, transform=transform)

    def DataLoader(self):
        train_data_loader = torch.utils.data.DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=self.mode)

        return train_data_loader