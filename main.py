import torch
import torchvision
from torch.autograd import Variable
from gan import Gen as generator
from gan import Discr as discriminator
from image_dataset import ImageDataset

batch_size = 1

def train_gan(generator, discriminator):
    # train discriminator with real and generator's fake data
    train_discriminator(generator, discriminator)
    # train generator using the result of the discriminator, but don't train the discrimininator
    train_generator(generator)

def train_discriminator(generator, discriminator, batch_size=100, epochs=50):
    optimizer = torch.optim.Adam(discriminator.parameters())
    loss_fun = discriminator.loss_fun()
    # teach the discriminator to distinguish between real and fake data
    real_image_dataset = torchvision.datasets.MNIST(
        './data', 
        download=True, 
        train=True, 
        transform=torchvision.transforms.ToTensor(),
    )
    training_dataset = ImageDataset(real_image_dataset, generator, batch_size)
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset, 
        shuffle=True, 
        batch_size=batch_size, 
        num_workers=2
    )
    for data in training_dataloader:
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = discriminator(inputs)
        loss = loss_fun(outputs, labels)
        print('discriminator loss: ' + str(loss.data[0]))
        loss.backward()
        optimizer.step()

            

def train_generator(generator):
    # generate a vector of random images
    optimizer = torch.optim.Adam(generator.parameters())
    fake_labels = torch.ones(batch_size)
    loss_fun = generator.loss_fun()


gen = generator()
dis = discriminator()
train_gan(gen, dis)