import sys
import random
import torch.optim as optim
import argparse
from pprint import pprint
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
import os
import cv2
import numpy as np
print(torch.__version__, torchvision.__version__)
from utils import label_to_onehot, cross_entropy_for_onehot, DatasetBD
from models.vision import LeNet

# Prepare arguments
parser = argparse.ArgumentParser()
# model
parser.add_argument('--cuda', type=int, default=1, help='cuda available')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--pretrained_model_path', type=str, default='./saved_models/pretrained_model.pth',
                    help='pretrained model path')
parser.add_argument('--target_model_path', type=str, default="./saved_models/target_trained_model.pth",
                    help='target model path')
# backdoor attacks
parser.add_argument('--target_label', type=int, default=9, help='class of target label')
parser.add_argument('--trigger_type', type=str, default='squareTrigger', help='type of backdoor trigger')
parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')
# Inversion algorithm
parser.add_argument('--algorithm', type=str, default="tmm", help='tmm, dlg, idlg')
parser.add_argument('--algorithm_epoch', type=int, default=240, help='algorithm epoch')
parser.add_argument('--algorithm_lr', type=int, default=0.1, help='algorithm leraning rate')
parser.add_argument('--result_save_epoch', type=int, default=40, help='algorithm epoch')
parser.add_argument('--weight', type=int, default=1, help='multiple of L2 distance difference')
parser.add_argument('--seed', type=int, default=1234, help='random seed number')
parser.add_argument('--directory', type=str, default='./result', help='algorithm result image save path')
args = parser.parse_args()

if __name__ == '__main__':

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print("Running on %s" % device)

    tp = transforms.ToTensor()
    tt = transforms.ToPILImage()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if args.dataset == 'CIFAR10':
        dst = datasets.CIFAR10("~/.torch",
                train=True,
                download=True,
                transform=transform)

    gt_data = dst[35][0].to(device)
    data = DatasetBD(args, gt_data,  transform=None, device=torch.device("cuda"), distance=1)
    gt_data =data.act().clone().detach().requires_grad_(True)
    r_label=dst[35][1]

    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = args.target_label * np.ones((gt_data.shape[0],), dtype=int)


    gt_label = torch.Tensor(gt_label).long().to(device)
    gt_label = gt_label.view(1, )

    gt_onehot_label = label_to_onehot(gt_label)
    criterion = cross_entropy_for_onehot

    # Load pre-training and target-training models
    net_before = LeNet().to(device)
    net_before.load_state_dict(torch.load(args.pretrained_model_path))

    net_after = LeNet().to(device)
    net_after.load_state_dict(torch.load(args.target_model_path))

    # Compute approximate gradient through parameter difference
    approx_gradients = []
    with torch.no_grad():
        for param_before, param_after in zip(net_before.parameters(), net_after.parameters()):
            approx_grad = param_after - param_before
            approx_gradients.append(approx_grad)

    original_dy_dx = approx_gradients

    # generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    #dummy_data = torch.ones(gt_data.size()).to(device).requires_grad_(True)

    # Algorithm parameter settings
    if args.algorithm == 'dlg':
        dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=args.algorithm_lr)
    if args.algorithm == 'idlg':
        dummy_label = gt_onehot_label
        optimizer = torch.optim.LBFGS([dummy_data], lr=args.algorithm_lr)
    if args.algorithm == 'tmm':
        dummy_label = gt_onehot_label
        optimizer = torch.optim.LBFGS([dummy_data], lr=1)

    # Inversion algorithm
    history = []
    args.weight = 1
    for iters in range(args.algorithm_epoch+1):
        args.iters=iters
        def closure():
            optimizer.zero_grad()
            dummy_pred = net_before(dummy_data)
            if args.algorithm == 'dlg':
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            if args.algorithm == 'idlg':
                dummy_onehot_label = dummy_label
            if args.algorithm == 'tmm':
                dummy_onehot_label = dummy_label
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net_before.parameters(), create_graph=True)

            # Reconstructed gradient and target gradient difference L2 norm
            grad_diff_l2 = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff_l2 += ((gx - gy) ** 2).sum()

            # TMM set the initial value of the gradient difference L2 norm
            if args.iters == 0 and args.algorithm == 'tmm':
                args.weight = float(90/grad_diff_l2)
            grad_diff_l2 = grad_diff_l2 * args.weight
            grad_diff_l2.backward()
            return grad_diff_l2

        optimizer.step(closure)
        if iters % args.result_save_epoch == 0:
            current_loss = closure()
            print(iters, "%.8f" % current_loss.item())
            history.append(tt(dummy_data[0].cpu()))

    plt.figure(figsize=(6, 6)).subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imshow(history[-1])
    plt.axis('off')
    #plt.show()
    if args.algorithm != 'tmm':
        filename = args.algorithm+'_example.png'
        directory = args.directory
        filepath = os.path.join('./result', filename)
        plt.savefig(filepath)
    else:
        filename = args.algorithm+'_preliminary_restored_example.png'
        filepath = os.path.join('./result', filename)
        plt.savefig(filepath)



    if args.algorithm == 'tmm' and args.trigger_type == 'squareTrigger':
        history.append(tt(dummy_data[0].cpu()))
        image = np.array(history[-1])
        # Apply bilateral filtering to optimize the image
        denoised_image = cv2.bilateralFilter(image, 9, 75, 75)

        plt.figure(figsize=(6, 6)).subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.imshow(denoised_image)
        plt.axis('off')
        filename = args.algorithm + '_optimization_example.png'
        filepath = os.path.join('./result', filename)
        plt.savefig(filepath)

        # Convert (H, W, C) shaped NumPy array to (C, H, W) shaped PyTorch tensor
        denoised_tensor = torch.from_numpy(denoised_image)
        denoised_tensor = denoised_tensor.permute(2, 0, 1)
        denoised_tensor = denoised_tensor.float() / 255.0

        with torch.no_grad():
            dummy_data[0] = denoised_tensor
        width = dummy_data.shape[3]
        height = dummy_data.shape[2]
        bestpic =dummy_data
        bestloss = current_loss

        distance = 1

        # Anchor point restore trigger
        for j in range(30):
            with torch.no_grad():
                for i in range(3):
                    dummy_data[0][i][width - 1-distance][height - 1-distance] = 0
                    dummy_data[0][i][width - 1-distance][height - 3-distance] = 0
                    dummy_data[0][i][width - 2-distance][height - 2-distance] = 0
                    dummy_data[0][i][width - 3-distance][height - 1-distance] = 0
                    dummy_data[0][i][width - 3-distance][height - 3-distance] = 0
                    if j == 19:
                        dummy_data[0][i][width - 1-distance][height - 2-distance] = 0
                        dummy_data[0][i][width - 2-distance][height - 1-distance] = 0
                        dummy_data[0][i][width - 2-distance][height - 3-distance] = 0
                        dummy_data[0][i][width - 3-distance][height - 2-distance] = 0
            optimizer = torch.optim.LBFGS([dummy_data], lr=1)
            history = []
            for iters in range(100+1):
                def closure():
                    optimizer.zero_grad()
                    dummy_pred = net_before(dummy_data)
                    dummy_onehot_label = dummy_label
                    dummy_loss = criterion(dummy_pred, dummy_onehot_label)
                    dummy_dy_dx = torch.autograd.grad(dummy_loss, net_before.parameters(), create_graph=True)

                    grad_diff_l2 = 0
                    for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                        grad_diff_l2 += ((gx - gy) ** 2).sum()

                    grad_diff_l2 = grad_diff_l2
                    grad_diff_l2.backward()
                    return grad_diff_l2

                optimizer.step(closure)
                if iters == 100:
                    current_loss = closure()
                    print("%.8f" % current_loss.item())
                    history.append(tt(dummy_data[0].cpu()))

            if (j+1)%5==0:
                plt.figure(figsize=(6, 6)).subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.imshow(history[-1])
                plt.axis('off')
                filename = args.algorithm + str(j) + '_optimization_example.png'
                filepath = os.path.join('./result', filename)
                plt.savefig(filepath)


