import torch
from my_dataloader import LoadDataset
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from vit import ViT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, required=True, default=28,
                    help='Size of the images')
parser.add_argument('--in_channels', type=int, required=True, default=1,
                    help='Channels of the images')
parser.add_argument('--emb_dim', type=int, required=True, default=768,
                    help='Hidden dimension of the ViT')
parser.add_argument('--patch_size', type=int, required=True, default=4,
                    help='Size of the patch')
parser.add_argument('--depth', type=int, required=True, default=4,
                    help='Number of ViT layers')
parser.add_argument('--num_heads', type=int, required=True, default=8,
                    help='Number of heads of ViT layers')
parser.add_argument('--mlp_ratio', type=int, required=True, default=4,
                    help='Ratio of hidden dims of MLP')
parser.add_argument('--epochs', type=int, required=True, default=10,
                    help='Training epochs')
parser.add_argument('--num_classes', type=int, required=True, default=10,
                    help='Number of classes for the classification task')
parser.add_argument('--dataset', type=str, default='MNIST',
                    help='Which dataset to train on our ViT')

args = parser.parse_args()
img_size = args.img_size
in_channels = args.in_channels
emb_dim = args.emb_dim
patch_size = args.patch_size
num_heads = args.num_heads
depth = args.depth
mlp_ratio = args.mlp_ratio
num_classes = args.num_classes
epochs = args.epochs
dataset_name = args.dataset


def test_per_epoch(model):
    model.eval()
    model.to(device)

    data = LoadDataset(dataset=dataset_name, mode="test", batch_size=32)
    data_loader = data.DataLoader()

    counts = 0
    correct = 0
    with torch.no_grad():
        for i, (image, label) in enumerate(data_loader):
            image, label = image.to(device), label.to(device)
            scores, attn = model(image)

            loss = criterion(scores, label)

            counts += label.size(0)
            correct += (torch.argmax(torch.nn.Softmax(-1)(scores), dim=-1) == label).sum().item()

        print("Test accuracy:", correct / counts, "| test loss:", loss.item())
        return correct / counts


device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = LoadDataset(dataset=dataset_name, batch_size=16)
data_loader = data.DataLoader()

model = ViT(in_channels, img_size, emb_dim, num_heads, patch_size, depth, mlp_ratio, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

running_loss = 0
i = 0
counts = 0
correct = 0
for epoch in range(epochs):
    correct = 0
    counts = 0
    for image, label in tqdm(data_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
        image, label = image.to(device), label.to(device)
        scores, _ = model(image)
        loss = criterion(scores, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        counts += label.size(0)
        correct += (torch.argmax(torch.nn.Softmax(-1)(scores), dim=-1) == label).sum().item()

    print("Epoch train accuracy:", correct / counts, "| train loss:", loss.item())
    test_acc = test_per_epoch(model)

torch.save({
    "img_size": img_size,
    "in_channels": in_channels,
    "emb_dim": emb_dim,
    "patch_size": patch_size,
    "num_heads": num_heads,
    "depth": depth,
    "mlp_ratio": mlp_ratio,
    "num_classes": num_classes,
    "epochs": epochs,
    "dataset_name": dataset_name,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': criterion,
}, f'./myvit_mnist.pth')
