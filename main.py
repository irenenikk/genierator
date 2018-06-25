import torch
import torchvision
from torch.autograd import Variable
from gan import Gen as generator
from gan import Discr as discriminator
from image_dataset import ImageDataset

def train_discriminator(real_inputs, real_labels, discriminator, loss_fun, batch_size):
    # discriminator and generator have separate optimizers
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    real_inputs, real_labels = Variable(real_inputs), Variable(real_labels)
    discriminator_optimizer.zero_grad()
    outputs = discriminator(real_inputs)
    discriminator_loss = loss_fun(outputs, real_labels)
    accuracy = calculate_accuracy(outputs, real_labels, batch_size)
    print('discriminator training accuracy: ' + str(accuracy))
    discriminator_loss.backward()
    discriminator_optimizer.step()

def calculate_accuracy(outputs, real_labels, batch_size):
    predictions = torch.round(outputs)
    correct = (predictions == real_labels).sum().float()
    accuracy = correct/batch_size
    return accuracy

def train_generator(batch_size, generator, discriminator, loss_fun):
    # we want the discriminator to think the fake data is real
    target_fake_labels = torch.ones(batch_size, 1)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    # generator's input
    random_seed = torch.randn(batch_size, generator.input_size)
    # discriminator and generator have separate optimizers
    generator_optimizer.zero_grad()
    fake_data = generator(random_seed)
    predictions = discriminator(fake_data)
    generator_loss = loss_fun(predictions, target_fake_labels)
    print('generator loss: ' + str(generator_loss))
    generator_loss.backward()
    generator_optimizer.step()

def train_gan(generator, discriminator, batch_size=100, epochs=10):
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
    )
    for e in range(epochs):
        # train for each data sample
        for data in training_dataloader:
            print('epoch:' + str(e))
            real_inputs, real_labels = data
            # first train discriminator with real data
            train_discriminator(real_inputs, real_labels, discriminator, loss_fun, batch_size)
            # then train generator with one batch
            train_generator(batch_size, generator, discriminator, loss_fun)

gen = generator()
dis = discriminator()
train_gan(gen, dis)