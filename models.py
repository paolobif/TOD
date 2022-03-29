from skimage import io, transform
import torch
import os
import cv2
import numpy as np

from torchvision import transforms, datasets, utils as vutils
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import pad


class WormClassifier(nn.Module):
    def __init__(self, dim=64):
        super(WormClassifier, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 12, 5, 1, padding=2)
        self.conv1_2 = nn.Conv2d(12, 12, 5, 1, padding=2)

        self.conv2_1 = nn.Conv2d(12, 24, 5, 1, padding=2)
        self.conv2_2 = nn.Conv2d(24, 24, 5, 1, padding=2)

        self.conv3_1 = nn.Conv2d(24, 36, 5, 1, padding=2)
        self.conv3_2 = nn.Conv2d(36, 36, 5, 1, padding=2)

        self.conv4_1 = nn.Conv2d(36, 48, 5, 1, padding=2)
        self.conv4_2 = nn.Conv2d(48, 48, 5, 1, padding=2)

        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x, features=False):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.max_pool2d(x, 2)

        xf = x
        x = torch.flatten(xf, 1)
        x_features = self.fc1(x)
        x_features = F.relu(x_features)
        x = torch.sigmoid(self.fc2(x_features))

        if features:
            return x, x_features

        return x


class SquarePad:
    def __init__(self, use=False):
        self.use = use

    def __call__(self, image):
        if not self.use:
            return image

        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return pad(image, padding, 0, 'constant')


class Pass:
    def __call__(self, image):
        return image


class WormDataLoader(Dataset):

    def __init__(self, path):
        self.path = path
        self.img_names = os.listdir(path)
        self.remove_ds()

        self.data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize((0.5), (0.5))
        ])

    def remove_ds(self):
        if '.DS_Store' in self.img_names:
            self.img_names.remove('.DS_Store')

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_path = os.path.join(self.path, self.img_names[index])
        image = io.imread(img_path)
        image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)

        image = self.transform(image)


# No mask version
def pre_process_img(img, mask_model=False):
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        SquarePad() if mask_model else Pass(),
        transforms.Resize((64, 64)),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize((0.5), (0.5))
    ])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    if mask_model:
        mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 11, 22)
        mask = cv2.bitwise_not(mask)
        img = cv2.bitwise_and(img, img, mask=mask)
    img = data_transform(img)
    img = img.unsqueeze(1)
    return img


# Auto Encoder Stuff
class ThreshMask:
    def __init__(self, use=False):
        self.use = use

    def __call__(self, image):
        if self.use:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
            mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 22)
            mask = cv2.bitwise_not(mask)
            image = cv2.bitwise_and(image, image, mask=mask)
        return image


def encoder_transofrm(img, mask=True):
    transformer = transforms.Compose([
            ThreshMask(use=mask),
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            SquarePad(use=mask),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(int(mean)), std=(int(std))),
            transforms.Resize((28, 28)),
        ])

    return transformer(img)
