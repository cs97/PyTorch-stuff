

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pandas, numpy, random
import matplotlib.pyplot as plt

import h5py

import time

import pickle

#==================================================================================
# check cuda
#==================================================================================
if torch.cuda.is_available():
  torch.set_default_tensor_type(torch.cuda.FloatTensor)
  print("using:", torch.cuda.get_device_name(0))
  pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device

#==================================================================================
# train data
#==================================================================================
input_file = 'celeba_aligned_small.h5py'
input_object = 'img_align_celeba'

#inputsize = [182, 182, 3]
inputsize = [218, 178, 3]
generatorinput = 100

#==================================================================================
# Test
#==================================================================================
def test_dataset():
    with h5py.File(input_file, 'r') as file_object:
        dataset = file_object['img_align_celeba']
        #dataset = file_object['img_gc']
        image = numpy.array(dataset['1.jpg'])
        plt.imshow(image, interpolation='none')
        plt.show()
        pass

    image.shape

def test_dataobject():
    dataset = Dataset(input_file)

    dataset.plot_image(0)

def test_dircriminator_train():
    D = Discriminator()
    D.to(device)

    for image_data_tensor in dataset:
        D.train(image_data_tensor, torch.cuda.FloatTensor([1.0])) #Eingabe

        D.train(generate_random_image((182*182*3)), torch.cuda.FloatTensor([0.0]))

        pass

def test_generator():
    G = Generator()
    G.to(device)

    output = G.forward(generate_random_seed(100))
    img = output.detach().cpu().numpy()
    plt.imshow(img, interpolation='none', cmap='Blues')
    plt.show()

#==================================================================================
# Dataset
#==================================================================================
class Dataset(Dataset):
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

  def plot_image(self, index):
    plt.imshow(numpy.array(self.dataset[str(index)+'.jpg']), interpolation='nearest')
    plt.show()
    pass
  pass

dataset = Dataset(input_file)

#==================================================================================
# Funktion
#==================================================================================
#class View(nn.Module):

def generate_random_image(size):
    random_data = torch.rand(size)
    return random_data

def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data

def test_randome_datatype():
    x =generate_random_image(2)
    print(x.device)

    y= generate_random_seed(2)
    print(y.device)

#==================================================================================
# Netz
#==================================================================================
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape= shape,

    def forward(self, x):
        return x.view(*self.shape)

class Discriminator(nn.Module):
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()

        # define neural network layers
        self.model = nn.Sequential(
            View(inputsize[0]*inputsize[1]*inputsize[2]),

            nn.Linear(inputsize[0]*inputsize[1]*inputsize[2], 100),
            #nn.Sigmoid(),
            #nn.LeakyReLU(0.02),
            nn.LeakyReLU(),

            nn.LayerNorm(100),

            nn.Linear(100, 1),
            nn.Sigmoid()
            #nn.LeakyReLU(0.02)

        )

        # create loss function
        #self.loss_function = nn.MSELoss()
        #S.96
        self.loss_function = nn.BCELoss()


        # create optimiser, simple stochastic gradient descent
        #self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)


        # counter and accumulator for progress
        self.counter = 0;
        self.progress = []

        pass


    def forward(self, inputs):
        # simply run model
        return self.model(inputs)


    def train(self, inputs, targets):
        # calculate the output of the network
        outputs = self.forward(inputs)

        # calculate loss
        loss = self.loss_function(outputs, targets)

        # increase counter and accumulate error every 10
        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)
            pass

        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass


    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        #df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        plt.show()
        pass

    pass

class Generator(nn.Module):
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()

        # define neural network layers
        self.model = nn.Sequential(

            nn.Linear(generatorinput, 3*10*10),
            #nn.Sigmoid(),

            #nn.LeakyReLU(0.02),
            nn.LeakyReLU(),

            nn.LayerNorm(3*10*10),

            nn.Linear(3*10*10, inputsize[0]*inputsize[1]*inputsize[2]),
            nn.Sigmoid(),
            #nn.LeakyReLU(0.02)
            View((inputsize[0], inputsize[1], inputsize[2]))

        )

        # create optimiser, simple stochastic gradient descent
        #self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)


        # counter and accumulator for progress
        self.counter = 0;
        self.progress = []

        pass


    def forward(self, inputs):
        # simply run model
        return self.model(inputs)


    def train(self, D, inputs, targets):
        # calculate the output of the network
        g_output = self.forward(inputs)

        # pass onto Discriminator
        d_output = D.forward(g_output)

        # calculate error
        loss = D.loss_function(d_output, targets)

        # increase counter and accumulate error every 10
        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass

        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass


    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        #df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        plt.show()
        pass

    pass

#==================================================================================
# Train
#==================================================================================
def train_gan():
    epochs = 1
    progress = 0

    for epoch in range(epochs):
        print ("epoch = ", epoch + 1)

        for image_data_tensor in dataset:
            D.train(image_data_tensor, torch.cuda.FloatTensor([1.0]))

            D.train(G.forward(generate_random_seed(generatorinput)).detach(), torch.cuda.FloatTensor([0.0]))

            G.train(D, generate_random_seed(generatorinput), torch.cuda.FloatTensor([1.0]))

            progress += 1
            if (progress % 500 == 0):
                print("progress = ", progress)

            #if (progress == 1000):
            #   break

        pass

#==================================================================================
# Progress
#==================================================================================
def print_progress():
    D.plot_progress()
    G.plot_progress()

#==================================================================================
# Result
#==================================================================================
def print_result():
    f, axarr = plt.subplots(2,3, figsize=(16,8))

    for i in range(2):
        for j in range(3):
            output = G.forward(generate_random_seed(generatorinput))
            img =output.detach().cpu().numpy()
            axarr[i,j].imshow(img, interpolation='none', cmap='Blues')
            pass
        pass
    plt.show()

def print_one_result():
    #f, axarr = plt.subplots(2,3, figsize=(16,8))
    output = G.forward(generate_random_seed(generatorinput))
    img =output.detach().cpu().numpy()
    #axarr[i,j].imshow(img, interpolation='none', cmap='Blues')
    plt.imshow(img, interpolation='none', cmap='Blues')
    plt.show()

#==================================================================================
# Main
#==================================================================================
def main():
    while True:

        print("1. generate gan")
        print("2. import gan")
        print("3. save gan")
        print("4. train once")
        print("5. print progress")
        print("6. print results")
        print("   exit")
        print("")

        cmd = input(">")

        if cmd == "1":
            D = Discriminator()
            D.to(device)
            G = Generator()
            G.to(device)

        if cmd == "2":
            D = torch.load("d.pt")
            G = torch.load("g.pt")

        if cmd == "3":
            torch.save(D, "d.pt")
            torch.save(G, "g.pt")

        if cmd == "4":
            start = time.time()
            train_gan()
            end = time.time()
            print('time: %f min' % ((end - start)/60))

        if cmd == "5":
            print_progress()

        if cmd == "6":
            print_result()

        if cmd == "exit":
            break


if __name__ == "__main__":
    main()







