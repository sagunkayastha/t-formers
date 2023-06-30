from PIL import Image, ImageFile
import torchvision.transforms as transforms
import torch.utils.data as data
from .image_folder import make_dataset
from util import task
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import h5py
import dask.array as da
from dask.distributed import Client, LocalCluster


class CreateDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        
        # self.img_paths, self.img_size = make_dataset(opt.img_file)
        # self.images, self.masks = h5_loader()
        self.images = np.load('/tng4/users/skayasth/Yearly/2023/June/MAT/train_images.npy')
        self.masks = np.load('/tng4/users/skayasth/Yearly/2023/June/MAT/train_masks.npy')
        self.img_size = self.images.shape[0]
        
        
    def __getitem__(self, index):
        img, img_path = self.images[index], './'
        mask = self.masks[index]     
        
        return {'img': img, 'img_path': img_path, 'mask': mask}

    def __len__(self):
        return self.img_size

    def name(self):
        return "inpainting dataset"

    

def dataloader(opt):
    datasets = CreateDataset(opt)
    dataset = data.DataLoader(datasets, batch_size=opt.batchSize, shuffle=not opt.no_shuffle, num_workers=int(opt.nThreads))

    return dataset

def h5_loader():
    # Set up a local cluster with multiple workers
    cluster = LocalCluster(n_workers=8)
    client = Client(cluster)

    # Open the HDF5 file
    with h5py.File('train_data_all.h5', 'r') as file:
        # Access the group containing the datasets
        group = file['data']

        # Load the datasets into dask arrays in parallel
        dask_arrays = {}
        for key in group.keys():
            dask_arrays[key] = da.from_array(group[key], chunks='auto')

        # Trigger computation and obtain the NumPy arrays
        numpy_arrays = da.compute(*dask_arrays.values())

    # Retrieve the NumPy arrays from the output
    

    # Close the Dask client and cluster
    client.close()
    cluster.close()
    return numpy_arrays

