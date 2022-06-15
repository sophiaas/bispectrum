import torch
from collections import OrderedDict
import numpy as np
from torch.utils.data import Dataset
import os


class TransformDataset:
    def __init__(self, dataset, transforms):
        """
        Arguments
        ---------
        dataset (obj):
            Object from patterns.natural or patterns.synthetic
        transforms (list of obj):
            List of objects from transformations. The order of the objects
            determines the order in which they are applied.
        """
        if type(transforms) != list:
            transforms = [transforms]
        self.transforms = transforms
        self.gen_transformations(dataset)
        if len(self.data.shape) == 3:
            self.img_size = tuple(self.data.shape[1:])
        else:
            self.dim = self.data.shape[-1]

    def gen_transformations(self, dataset):
        transform_dict = OrderedDict()
        transformed_data = dataset.data.clone()
        new_labels = dataset.labels.clone()
        for transform in self.transforms:
            transformed_data, new_labels, transform_dict, t = transform(
                transformed_data, new_labels, transform_dict
            )
            transform_dict[transform.name] = t
        self.data = transformed_data
        self.labels = new_labels
        self.transform_labels = transform_dict

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
    
    
class VanHateren(Dataset):
    def __init__(
        self,
        path="datasets/van-hateren/",
        normalize=True,
        select_img_path="select_imgs.txt",
        patches_per_image=10,
        patch_size=16,
        min_contrast=0.1,
    ):


        super().__init__()
            
        self.name = "van-hateren"
        self.dim = patch_size ** 2
        self.path = path
        self.patches_per_image = patches_per_image
        self.select_img_path = select_img_path
        self.normalize = normalize
        self.patch_size = patch_size
        self.min_contrast = min_contrast
        self.img_shape = (1024, 1536)
        
        full_images = self.load_images()

        self.data, self.labels = self.get_patches(full_images)

        
    def get_patches(self, full_images):
        data = []
        labels = []

        i = 0
        
        for img in full_images:
            for p in range(self.patches_per_image):
                low_contrast = True
                j = 0 
                while low_contrast and j < 100:
                    start_x = np.random.randint(0, self.img_shape[1] - self.patch_size)
                    start_y = np.random.randint(0, self.img_shape[0] - self.patch_size)
                    patch = img[
                        start_y : start_y + self.patch_size, start_x : start_x + self.patch_size
                    ]
                    if patch.std() >= self.min_contrast:
                        low_contrast = False
                        data.append(patch)
                        labels.append(i)
                    j += 1
                
                if j == 100 and not low_contrast:
                    print("Couldn't find patch to meet contrast requirement. Skipping.")
                    continue

                i += 1
        data = torch.tensor(np.array(data))
        labels = torch.tensor(np.array(labels))
        return data, labels
                        
        
    def load_images(self):
        if self.select_img_path is not None:
            with open(self.path + self.select_img_path, "r") as f:
                img_paths = f.read().splitlines()
        else:
            img_paths = os.listdir(path + "images/")

        all_imgs = []

        for i, img_path in enumerate(img_paths):
            try:
                with open(self.path + "images/" + img_path, 'rb') as handle:
                    s = handle.read()
            except:
                print("Can't load image at path {}".format(self.path + img_path))
                continue
            img = np.fromstring(s, dtype='uint16').byteswap()
            if self.normalize:
                # Sets image values to lie between 0 and 1
                img = img.astype(float)
                img -= img.min()
                img /= img.max()
                img -= img.mean()
                img *= 2
            img = img.reshape(self.img_shape)
            all_imgs.append(img)

        all_imgs = np.array(all_imgs)
        return all_imgs

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
    
