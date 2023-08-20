import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Define the Generator and Discriminator networks
class Generator(nn.Module):

    def __init__(self, latent_dim, image_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, image_dim),
            nn.Tanh()
        )


    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, image_dim):
        super(Discriminator, self).__init__()
        self.image_dim = image_dim
        self.model = nn.Sequential(
            nn.Linear(image_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):

        return self.model(img)


# Function to train the GAN
def train_gan(generator, discriminator, dataloader, num_epochs, latent_dim, device):
    criterion = nn.BCELoss()
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    for epoch in range(num_epochs):
        for batch_idx, (real_images, _) in enumerate(dataloader):
            real_images = real_images.view(-1, 784).to(device)
            real_labels = torch.ones(real_images.size(0), 1).to(device)
            fake_labels = torch.zeros(real_images.size(0), 1).to(device)

            # Train Discriminator
            d_optimizer.zero_grad()
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)
            d_loss_real.backward()

            z = torch.randn(real_images.size(0), latent_dim).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss_fake.backward()

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            z = torch.randn(real_images.size(0), latent_dim).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], D_Loss: {d_loss.item():.4f}, G_Loss: {g_loss.item():.4f}")

# Function to generate new images using the trained generator
def generate_images(generator, num_images, latent_dim, device):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(device)
        generated_images = generator(z)
    return generated_images.cpu()


if __name__ == '__main__':
    # Set hyperparameters
    num_epochs = 50
    batch_size = 64
    latent_dim = 100
    image_dim = 28 * 28
    specified_digit = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    digit_indices = [i for i in range(len(train_dataset.targets)) if train_dataset.targets[i] == specified_digit]
    train_dataset.data = train_dataset.data[digit_indices]
    train_dataset.targets = [train_dataset.targets[i] for i in digit_indices]

    # Create GAN and move to device
    generator = Generator(latent_dim, image_dim).to(device)
    discriminator = Discriminator(image_dim).to(device)

    # Train the GAN
    train_gan(generator, discriminator, dataloader, num_epochs, latent_dim, device)

    # Generate new images using the trained generator
    num_images_to_generate = 2
    generated_images = generate_images(generator, num_images_to_generate, latent_dim, device)

    # Display the generated images
    fig, axes = plt.subplots(1, num_images_to_generate, figsize=(10, 2))
    for i in range(num_images_to_generate):
        axes[i].imshow(generated_images[i].view(28, 28).numpy(), cmap='gray')
        axes[i].axis('off')
    plt.show()






