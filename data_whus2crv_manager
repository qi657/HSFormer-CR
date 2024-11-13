# -- coding: utf-8 --
# @Time : 2024/11/13 9:29
# @Author : Stephanie
# @Email : sunc696@gmail.com
# @File : data_whus2crv_manager.py

from __future__ import absolute_import, division, print_function, unicode_literals
from WHUS2_CRv.gdaldiy import *
import glob
import os

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def diydecay(steps, baselr, cycle_step=100000, decay_steps=100, decay_rate=0.98):
    n = steps // cycle_step
    clr = baselr * (0.96 ** n)
    steps = steps - n * cycle_step
    k = steps // decay_steps
    dlr = clr * (decay_rate ** k)
    return dlr


def decay(global_steps, baselr, start_decay_step=100000, cycle_step=100000, decay_steps=100, decay_rate=0.98):
    lr = np.where(np.greater_equal(global_steps, start_decay_step),
                  diydecay(global_steps - start_decay_step, baselr, cycle_step, decay_steps, decay_rate),
                  baselr)
    return lr


def make_train_data_list(data_path):  
    filepath = glob.glob(os.path.join(data_path, "*"))  
    image_path_lists = []
    for i in range(len(filepath)):
        path = glob.glob(os.path.join(filepath[i], "*"))
        for j in range(len(path)):
            image_path_lists.append(path[j]) 
    return image_path_lists


def liner_2(input_):  
    def strech(img):
        low, high = np.percentile(img, (2, 98))
        img[low > img] = low
        img[img > high] = high
        return (img - low) / (high - low + 1e-10)

    if len(input_.shape) > 2:
        for i in range(input_.shape[-1]):
            input_[:, :, i] = strech(input_[:, :, i])
    else:
        input_ = strech(input_)
    return input_


def get_write_picture(row_list):  
    row_ = []
    for i in range(len(row_list)):
        row = row_list[i]
        col_ = []
        for image in row:
            x_image = image[:, :, [2, 1, 0]]
            if i < 1:
                x_image = liner_2(x_image)
            col_.append(x_image)
        row_.append(np.concatenate(col_, axis=1))
    if len(row_list) == 1:
        output = np.concatenate(col_, axis=1)
    else:
        output = np.concatenate(row_, axis=0)  
    return output * 255


def randomflip(input_, n):
    return input_


def read_img(datapath, scale=255):
    img = imgread(datapath)  
    img = np.clip(img, 0, scale)  
    img = img / scale  
    return img


def read_imgs(datapath, scale=255, k=2):
    img_list = []
    for path in datapath:
        img = read_img(path, scale)
        img = randomflip(img, k)  
        img = np.expand_dims(img, axis=0) 
        img_list.append(img)

    imgs = np.concatenate(img_list, axis=0)
    imgs = torch.tensor(imgs, dtype=torch.float32)
    return imgs

class UnifiedDataset(Dataset):
    def __init__(self, file_list_10m, file_list_20m, file_list_60m, scale=10000):
        assert len(file_list_10m) == len(file_list_20m) == len(file_list_60m), \
            "10m, 20m, and 60m file lists must have the same length"

        self.file_list_10m = file_list_10m
        self.file_list_20m = file_list_20m
        self.file_list_60m = file_list_60m
        self.scale = scale

    def __len__(self):
        return len(self.file_list_10m)

    def __getitem__(self, idx):
        # Load 10m image (cloud and clear)
        cloud_file_10m, clear_file_10m = self.file_list_10m[idx]
        cloud_img_10m = read_img(cloud_file_10m, self.scale)
        clear_img_10m = read_img(clear_file_10m, self.scale)

        # Load 20m image (cloud and clear)
        cloud_file_20m, clear_file_20m = self.file_list_20m[idx]
        cloud_img_20m = read_img(cloud_file_20m, self.scale)
        clear_img_20m = read_img(clear_file_20m, self.scale)

        # Load 60m image (cloud and clear)
        cloud_file_60m, clear_file_60m = self.file_list_60m[idx]
        cloud_img_60m = read_img(cloud_file_60m, self.scale)
        clear_img_60m = read_img(clear_file_60m, self.scale)

        # Convert images to tensors
        cloud_tensor_10m = torch.tensor(cloud_img_10m, dtype=torch.float32)
        clear_tensor_10m = torch.tensor(clear_img_10m, dtype=torch.float32)

        cloud_tensor_20m = torch.tensor(cloud_img_20m, dtype=torch.float32)
        clear_tensor_20m = torch.tensor(clear_img_20m, dtype=torch.float32)

        cloud_tensor_60m = torch.tensor(cloud_img_60m, dtype=torch.float32)
        clear_tensor_60m = torch.tensor(clear_img_60m, dtype=torch.float32)

        # Return all images as a tuple
        return (cloud_tensor_10m, clear_tensor_10m,
                cloud_tensor_20m, clear_tensor_20m,
                cloud_tensor_60m, clear_tensor_60m)


def iterate_paired_img(file_list_10m, file_list_20m, file_list_60m, batch_size=1, scale=10000, num_workers=0):
    # Initialize the paired dataset and dataloader
    dataset = UnifiedDataset(file_list_10m, file_list_20m, file_list_60m, scale)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=2)
    return dataloader

