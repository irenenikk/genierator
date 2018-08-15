import torch, torchvision
from torch.utils.data import Dataset
import random
import numpy as np
import matplotlib.image as img
import matplotlib.image as img
import glob, os

class CatImageDataset(Dataset):
  def __init__(self, batch_size):
    self.batch_size = batch_size
    self.real_data_shape = (100, 100, 3)
    data = []
    for dir in glob.glob('./data/cats/CAT*'):
      for file in glob.glob(os.path.join(dir, '*.jpg')):
        image = img.imread(file)
        image_np = np.array(image)
        image_tensor = torch.from_numpy(image_np)
        transform = self.image_transform()
        image = transform(image)
        data.append((image, 'cat'))
    self.data = data

  def __len__(self):
    return len(self.data)
  
  def image_transform(self):
    return torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(self.real_data_shape[0:2]),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor()
    ])

  def __getitem__(self, i):
    return self.data[i]
    
