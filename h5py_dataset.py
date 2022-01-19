#!/bin/python3
#
import torch
import torch.nn as nn
import pandas, numpy, random

import h5py
from torch.utils.data import Dataset

input_file = ''
input_object = ''

#==================================================================================
# Dataset
#==================================================================================
class InputDataset(Dataset):
  def __init__(self, file):
    self.file_object = h5py.File(file, 'r')
    self.dataset = self.file_object[input_object]
    pass

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    if (index >= len(self.dataset)):
      raise IndexError()
    img = numpy.array(self.dataset[str(index)+'.jpg'])
    return torch.cuda.FloatTensor(img) / 255.0
    #return torch.FloatTensor(img) / 255.0

  def plot_image(self, index):
    plt.imshow(numpy.array(self.dataset[str(index)+'.jpg']), interpolation='nearest')
    plt.show()
    pass
  pass

#dataset = InputDataset(input_file)

def get_dataset():
    return InputDataset(input_file)
