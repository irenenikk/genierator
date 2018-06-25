import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
  def __init__(self, real_dataset, generator, batch_size):
    self.real_dataset = real_dataset
    self.generator = generator
    self.batch_size = batch_size

  def __len__(self):
    return 2 * len(self.real_dataset)
  
  def __getitem__(self, i):
    if i < len(self.real_dataset):
      # the data has to be reshaped to a vector
      real_data, _ = self.real_dataset[i]
      real_data = real_data.view(real_data.size(0), -1)
      return (real_data, torch.ones(1))
    else:
      random_seed = torch.randn(1, self.generator.input_size)
      # fake data label is 0
      fake_data = self.generator(random_seed)
      return (fake_data.data, torch.zeros(1))
