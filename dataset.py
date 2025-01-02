import os
from torch.utils.data import Dataset
from PIL import Image


class ImagetoImageDataset(Dataset):
    def __init__(self, domainA_dir, domainB_dir,domainC_dir, transform=None):
        self.domainA_images = [os.path.join(domainA_dir, x) for x in os.listdir(domainA_dir) if x.endswith('.png') or x.endswith('.jpg')]
        self.domainB_images = [os.path.join(domainB_dir, x) for x in os.listdir(domainB_dir) if x.endswith('.png') or x.endswith('.jpg')]
        self.domainC_images = [os.path.join(domainC_dir, x) for x in os.listdir(domainC_dir) if x.endswith('.png') or x.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return min(len(self.domainA_images), len(self.domainB_images),len(self.domainC_images))

    def __getitem__(self, idx):
        img_A = Image.open(self.domainA_images[idx]).convert('RGB')
        img_B = Image.open(self.domainB_images[idx]).convert('RGB')
        img_C = Image.open(self.domainC_images[idx]).convert('RGB')
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
            img_C = self.transform(img_C)

        return img_A, img_B, img_C
