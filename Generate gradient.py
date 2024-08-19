import sys
import os
import random
import torch.optim as optim
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
import scipy.stats
import cv2
import numpy as np
print(torch.__version__, torchvision.__version__)
from utils import label_to_onehot, cross_entropy_for_onehot, DatasetBD

# Prepare arguments
parser = argparse.ArgumentParser()
# model
parser.add_argument('--cuda', type=int, default=1, help='cuda available')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
# backdoor attacks
parser.add_argument('--target_label', type=int, default=9, help='class of target label')
parser.add_argument('--trigger_type', type=str, default='squareTrigger', help='type of backdoor trigger')
parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')
# Inversion algorithm
parser.add_argument('--algorithm', type=str, default="tmm", help='tmm, dlg, idlg')
parser.add_argument('--algorithm_epoch', type=int, default=240, help='algorithm epoch')
parser.add_argument('--algorithm_lr', type=int, default=1, help='algorithm leraning rate')
parser.add_argument('--result_save_epoch', type=int, default=40, help='algorithm epoch')
parser.add_argument('--weight', type=int, default=1, help='Multiple of L2 distance difference')
parser.add_argument('--seed', type=int, default=1234, help='Random seed')
parser.add_argument('--directory', type=str, default='./result', help='algorithm result image save path')
args = parser.parse_args()

if __name__ == '__main__':

    # Set device to CPU by default
    device = "cpu"
    # Check if CUDA is available and switch device to GPU if it is
    if torch.cuda.is_available():
        device = "cuda"
    print("Running on %s" % device)

    # Define transformations for the dataset
    tt = transforms.ToPILImage()  # Convert tensors to PIL images

    # Compose transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensor format
    ])

    # Load CIFAR-10 dataset if specified
    if args.dataset == 'CIFAR10':
        dst = datasets.CIFAR10("~/.torch",
                               train=True,
                               download=True,
                               transform=transform)

    # Get a sample image from the dataset and move it to the specified device
    gt_data = dst[35][0].to(device)
    data = DatasetBD(args, gt_data, transform=None, device=torch.device("cuda"), distance=1)

    # Convert the backdoor data to a tensor that requires gradients
    gt_data = data.act().clone().detach().requires_grad_(True)
    r_label = dst[35][1]

    # Reshape the data for the model
    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = args.target_label * np.ones((gt_data.shape[0],), dtype=int)

    # Convert labels to tensor format and move to the specified device
    gt_label = torch.Tensor(gt_label).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label)  # Convert labels to one-hot encoding

    # Display the original image
    plt.figure(figsize=(6, 6)).subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imshow(tt(gt_data[0].cpu()))  # Convert tensor back to PIL image for display
    plt.axis('off')
    filename = 'preliminary_example.png'
    directory = args.directory
    filepath = os.path.join(directory, filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)

    from models.vision import LeNet

    # Initialize the model and move it to the specified device
    net = LeNet().to(device)
    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)

    # Load pretrained model parameters
    loaded_params = torch.load("./saved_models/pretrained_model.pth")
    net.load_state_dict(loaded_params)

    # Define loss function and optimizer
    criterion = cross_entropy_for_onehot
    optimizer = optim.SGD(net.parameters(), lr=0.001, weight_decay=0.9)

    # Set model to training mode
    net.train()

    # Compute original gradients
    pred = net(gt_data)
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    # Update model parameters with original gradients
    with torch.no_grad():
        for param, grad in zip(net.parameters(), original_dy_dx):
            param.add_(grad)

    # Save the updated model parameters
    torch.save(net.state_dict(), "./saved_models/target_trained_model.pth")
    print('Model saved')