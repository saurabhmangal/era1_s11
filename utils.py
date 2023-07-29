from __future__ import print_function
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import random
from torch_lr_finder import LRFinder
from torch.optim.lr_scheduler import OneCycleLR

import warnings
warnings.filterwarnings('ignore')


class Cifar10SearchDataset(datasets.CIFAR10):

    def __init__(self, root="~/data", train=True, download=True, transform=None):

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):

        image, label = self.data[index], self.targets[index]

        if self.transform is not None:

            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


def create_train_data_loader(train, **dataloader_args):
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
    return train_loader


def create_test_data_loader(test, **dataloader_args):
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
    return test_loader


def train_transforms(means, stds):
    transforms = A.Compose(
        [
            A.Normalize(mean=means, std=stds, always_apply=True),
            A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
            A.RandomCrop(height=32, width=32, always_apply=True),
            A.HorizontalFlip(),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=means),
            ToTensorV2(),
        ]
    )
    return transforms


def test_transforms(means, stds):
    transforms = A.Compose(
        [
            A.Normalize(mean=means, std=stds, always_apply=True),
            ToTensorV2(),
        ]
    )
    return transforms


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def display_mis_images(miss_classified_data, n_images):
    random_images = range(0, len(miss_classified_data[0]))
    random_selects = random.sample(random_images, n_images)

    fig_miss_class = plt.figure()
    count = 0
    for i in random_selects:
        plt.subplot(2, int(n_images/2), count+1)
        plt.imshow(miss_classified_data[2][i].cpu().permute(1, 2, 0))
        plt.title("p=" + str(miss_classified_data[1][i]) + "   " + "a=" + str(miss_classified_data[0][i]))
        plt.xticks([])
        plt.yticks([])
        count += 1

    return fig_miss_class


def learning_r_finder(m, optimizer, criterion, device, train_loader, n_iters=200, end_lr=10):
    lr_finder = LRFinder(m, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=n_iters, step_mode="exp")
    fig = lr_finder.plot()  # to inspect the loss-learning rate graph
    lr_finder.reset()


def OneCycleLR_policy(optimizer, train_loader, EPOCHS, peak_value=5.0, div_factor=100, final_div_factor=100,max_lr=1.59E-03):
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=peak_value/EPOCHS,
        div_factor=div_factor,
        three_phase=False,
        final_div_factor=final_div_factor,
        anneal_strategy='linear',
    )
    return scheduler
