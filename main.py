import torch
import torchvision
from torch.autograd import Variable
from gan import Gen as generator
from gan import Discr as discriminator
from image_dataset import ImageDataset
import matplotlib.pyplot as plt

def train_discriminator(real_inputs, real_labels, discriminator, loss_fun, batch_size):
    '''
        Train the discriminator with one batch of mixed real and fake data
    '''
    # discriminator and generator have separate optimizers
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    real_inputs, real_labels = Variable(real_inputs), Variable(real_labels)
    discriminator_optimizer.zero_grad()
    outputs = discriminator(real_inputs)
    discriminator_loss = loss_fun(outputs, real_labels)
    accuracy = calculate_accuracy(outputs, real_labels, batch_size)
    print('discriminator training accuracy: ' + str(accuracy.item()))
    discriminator_loss.backward()
    discriminator_optimizer.step()

def calculate_accuracy(outputs, real_labels, batch_size):
    '''
        Calculate training accuracy
    '''
    predictions = torch.round(outputs)
    correct = (predictions == real_labels).sum().float()
    accuracy = correct/batch_size
    return accuracy

def train_generator(batch_size, generator, discriminator, loss_fun):
    '''
        Train the generator on one batch using the discriminator. 
        Generate a batch of fake pictures, classify them with
        the discriminator and calculate loss based on how sure the 
        discriminator was that the fake data was real.
    '''
    # we want the discriminator to think the fake data is real
    target_fake_labels = torch.ones(batch_size, 1)
    # discriminator and generator have separate optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    # generator's input
    random_seed = torch.randn(batch_size, generator.input_size)
    generator_optimizer.zero_grad()
    # generate a batch of fake data
    fake_data = generator(random_seed)
    # see what the discriminator thinks it is
    predictions = discriminator(fake_data)
    # we wnat the discriminator to think they're real
    generator_loss = loss_fun(predictions, target_fake_labels)
    print('generator loss: ' + str(generator_loss.item()))
    generator_loss.backward()
    generator_optimizer.step()

def train_gan(generator, discriminator, batch_size=256, epochs=2):
    '''
        The training loop for the whole network
    '''
    loss_fun = discriminator.loss_fun()
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
        print('epoch:' + str(e))
        for data in training_dataloader:
            real_inputs, real_labels = data
            # first train discriminator with real data
            train_discriminator(real_inputs, real_labels, discriminator, loss_fun, batch_size)
            # then train generator with one batch
            train_generator(batch_size, generator, discriminator, loss_fun)
    return training_dataset

def test_generator(gen, real_data_shape):
    fig = plt.figure(figsize=(8, 8))
    cols, rows = 4, 4
    for i in range(1, cols * rows + 1):
        # get images from trained generator
        random_seed = torch.randn(1, gen.input_size)
        image = gen(random_seed)
        # reshape back to original form
        image = image.view(real_data_shape[1], real_data_shape[2])
        image = image.detach().numpy()
        fig.add_subplot(rows, cols, i)
        plt.imshow(image)
    plt.show()


gen = generator()
dis = discriminator()
real_image_dataset = train_gan(gen, dis)
# generate example images
test_generator(gen, real_image_dataset.real_data_shape)
