import torch
from torch.utils.data import Dataset
import random
import numpy as np

class ImageDataset(Dataset):
  def __init__(self, real_dataset, generator, batch_size):
    self.real_dataset = real_dataset
    self.generator = generator
    self.batch_size = batch_size
    self.real_data_shape = None

  def __len__(self):
    return 2 * len(self.real_dataset)

  def get_random_tenosor_in_range(self, start, finish):
    num = random.uniform(start, finish)
    num_array = np.array([num])
    return torch.from_numpy(num_array).view(1, 1).float()
  
  def __getitem__(self, i):
    if i < len(self.real_dataset):
      # the data has to be reshaped to a vector
      real_data, _ = self.real_dataset[i]
      if (self.real_data_shape == None):
        self.real_data_shape = real_data.size()
      real_data = real_data.view(real_data.size(0), -1)
      return (real_data, self.get_random_tenosor_in_range(0.7, 1.2))
    else:
      random_seed = torch.randn(1, self.generator.input_size)
      # fake data label is 0
      fake_data = self.generator(random_seed)
      return (fake_data.data, self.get_random_tenosor_in_range(0.0, 0.3))
