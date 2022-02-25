from skimage import io, transform
import torch
import os
import cv2

from torchvision import transforms, datasets, utils as vutils
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


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


def pre_process_img(img):
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize((0.5), (0.5))
    ])

    img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    img = data_transform(img)
    img = img.unsqueeze(1)
    return img
