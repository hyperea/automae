import h5py
import numpy as np
import os
from tqdm import tqdm

from torchvision.datasets import ImageFolder

IMAGENET_FOLDER = "/data/ILSVRC2012"

hf = h5py.File(os.path.join(IMAGENET_FOLDER, "ImageNet.hdf5"), 'a')

val = hf.create_group("val")
val_path = os.path.join(IMAGENET_FOLDER, "val")
for klass in tqdm(os.listdir(val_path)):
    class_path = os.path.join(val_path, klass)
    group = val.create_group(klass)
    for file in os.listdir(class_path):
        file_path = os.path.join(class_path, file)
        with open(file_path, "rb") as img_f:
            binary_data = img_f.read()
        binary_data_np = np.asarray(binary_data)
        dset = group.create_dataset(file, data=binary_data_np)


train = hf.create_group("train")
train_path = os.path.join(IMAGENET_FOLDER, "train")

for klass in tqdm(os.listdir(train_path)):
    class_path = os.path.join(train_path, klass)
    group = train.create_group(klass)
    for file in os.listdir(class_path):
        file_path = os.path.join(class_path, file)
        with open(file_path, "rb") as img_f:
            binary_data = img_f.read()
        binary_data_np = np.asarray(binary_data)
        dset = group.create_dataset(file, data=binary_data_np)


hf.close()