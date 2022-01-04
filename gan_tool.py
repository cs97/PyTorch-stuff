#!/bin/python3
#
#pip install pandas
#pip install numpy
#pip install matplotlib
#pip install h5py

import torch
import torch.nn as nn
import pandas, numpy, random
import matplotlib.pyplot as plt
import time

#==================================================================================
# check cuda
#==================================================================================
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print("using:", torch.cuda.get_device_name(0))
    pass
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#==================================================================================
# netz
#==================================================================================
inputsize = [218, 178, 3]
generatorinput = 100

import netz

#==================================================================================
# Test
#==================================================================================
#import test

#==================================================================================
# Dataset
#==================================================================================

import h5py_dataset

h5py_dataset.input_file =  'celeba_aligned_small.h5py'
h5py_dataset.input_object = 'img_align_celeba'

dataset = h5py_dataset.get_dataset()

#==================================================================================
# Funktion
#==================================================================================

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
            if (progress % 5000 == 0):
                print("[+] progress = ", progress, "/", len(dataset))
            if (D.counter % 10000 == 0):
                print("[+] overall progress = ", D.counter)

            if (progress == 2000):
               break
        pass

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

    global D
    global G
    while True:
        cmd = input("gan-tool > ")

        if cmd == "help":
            print("")
            print("gen\t\tgenerate gan")
            print("import\t\timport gan")
            print("save\t\tsave gan")
            print("train\t\ttrain once")
            print("progress\tprint progress")
            print("results\t\tprint results")
            print("exit")

        if cmd == "gen":
            netz.inputsize = inputsize
            netz.generatorinput = generatorinput

            D = netz.Discriminator2()
            D.to(device)
            G = netz.Generator()
            G.to(device)

        if cmd == "import":
            D = torch.load("d.pt")
            G = torch.load("g.pt")

        if cmd == "save":
            torch.save(D, "d.pt")
            torch.save(G, "g.pt")

        if cmd == "train":
            start = time.time()
            train_gan()
            end = time.time()
            print('time: %f min' % ((end - start)/60))

        if cmd == "progress":
            D.plot_progress()
            G.plot_progress()

        if cmd == "results":
            print_result()

        if cmd == "exit":
            break




def menu():
    cmd = input("gan-tool > ")
    #cmd = "help"

    match cmd:

        case "gen":
            netz.inputsize = inputsize
            netz.generatorinput = generatorinput
            D = netz.Discriminator2()
            D.to(device)
            G = netz.Generator()
            G.to(device)

        case "import":
            D = torch.load("d.pt")
            G = torch.load("g.pt")

        case "save":
            torch.save(D, "d.pt")
            torch.save(G, "g.pt")

        case "train":
            start = time.time()
            train_gan()
            end = time.time()
            print('time: %f min' % ((end - start)/60))

        case "progress":
            D.plot_progress()
            G.plot_progress()

        case "results":
            print_result()

        case "exit":
            return

        case _:
            print("")
            print("gen\t\tgenerate gan")
            print("import\t\timport gan")
            print("save\t\tsave gan")
            print("train\t\ttrain once")
            print("progress\tprint progress")
            print("results\t\tprint results")
            print("exit")


if __name__ == "__main__":
    #main()
    menu()






