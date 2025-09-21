import imageio
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

import matplotlib

matplotlib.style.use("ggplot")
torch.set_float32_matmul_precision("high")


# @TimeMeasure
@njit(parallel=True)
def run_loop(data, output_array, number_bins, index, bin_length=4096):
    """
    Function to loop through the data and output the data into the output_array

    Parameters:
    - data: data to loop through
    - output_array: array to output the data to
    - number_bins: number of bins
    - bin_length: length of each bin

    Returns:
    - None
    """

    for j in prange(number_bins):  # Use prange for parallelization
        output_array[j, index, :, :] = data[:, j * bin_length : (j + 1) * bin_length]


@jit(nopython=True)
def final_loss(bce_loss, mu, logvar):
    """
    Function to calculate the final loss for the VAE model.

    Parameters:
    - bce_loss: the binary cross-entropy loss
    - mu: the mean from the encoder
    - logvar: the log variance from the encoder

    Returns:
    - final_loss: the combined loss of BCE and KLD
    """

    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(model, dataloader, dataset, device, optimizer, criterion):
    """
    Function to train the VAE model.

    Parameters:
    - model: the VAE model to train
    - dataloader: the DataLoader for the training dataset
    - dataset: the training dataset
    - device: the device to train on (CPU or GPU)
    - optimizer: the optimizer for training
    - criterion: the loss function to use

    Returns:
    - running_loss: the average loss over the training epoch
    """
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(
        enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)
    ):
        counter += 1
        if isinstance(data, (tuple, list)):
            data = data[0]

        data = data.to(device)  # send to GPU

        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / counter


def validate(model, dataloader, dataset, device, criterion):
    """
    Function to validate the VAE model.

    Parameters:
    - model: the VAE model to validate
    - dataloader: the DataLoader for the validation dataset
    - dataset: the validation dataset
    - device: the device to validate on (CPU or GPU)
    - criterion: the loss function to use

    Returns:
    - running_loss: the average loss over the validation dataset
    - recon_images: the reconstructed images from the last batch
    """

    model.eval()
    running_loss = 0.0
    counter = 0
    recon_images = None
    with torch.no_grad():
        for i, data in tqdm(
            enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)
        ):
            counter += 1
            if isinstance(data, (tuple, list)):
                data = data[0]

            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()

            # Save last batch of reconstructions
            if i == int(len(dataset) / dataloader.batch_size) - 1:
                recon_images = reconstruction

    return running_loss / counter, recon_images
