#!/bin/python3

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pandas, numpy, random
import matplotlib.pyplot as plt

inputsize = [0, 0, 0]
generatorinput = 0


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
        super().__init__()

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


        #self.loss_function = nn.MSELoss()
        self.loss_function = nn.BCELoss()

        #self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

        self.counter = 0;
        self.progress = []

        pass


    def forward(self, inputs):
        return self.model(inputs)


    def train(self, inputs, targets):
        outputs = self.forward(inputs)

        loss = self.loss_function(outputs, targets)

        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 10000 == 0):
            print("[+] overall progress = ", self.counter)
            pass

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass


    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        #df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        #plt.show()
        plt.savefig('progress-D-png')
        pass

    pass

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

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

        #self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

        self.counter = 0;
        self.progress = []

        pass


    def forward(self, inputs):
        return self.model(inputs)


    def train(self, D, inputs, targets):
        g_output = self.forward(inputs)

        d_output = D.forward(g_output)

        loss = D.loss_function(d_output, targets)

        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        #df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        #plt.show()
        plt.savefig('progress-G.png')
        pass

    pass



#==================================================================================
# Netz template
#==================================================================================


class template_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.counter = 0;
        self.progress = []
        """
        self.model = nn.Sequential(
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

        #self.loss_function = nn.MSELoss()
        #self.loss_function = nn.BCELoss()

        #self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        #self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)
        """
        pass

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)
        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        pass

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        #df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        #plt.show()
        plt.savefig('progress.png')
        pass
    pass







class Discriminator2(template_net):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            View(inputsize[0]*inputsize[1]*inputsize[2]),
            nn.Linear(inputsize[0]*inputsize[1]*inputsize[2], 100),
            nn.LeakyReLU(),
            nn.LayerNorm(100),
            nn.Linear(100, 1),
            nn.Sigmoid()
            )
        self.loss_function = nn.BCELoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)




