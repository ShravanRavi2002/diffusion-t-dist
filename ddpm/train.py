import torch
import tqdm
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn

from ddpm import utils
from ddpm.ddpm import DDPM
from ddpm.unet import UNet
from ddpm.utils import show_images, generate_new_images
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
import matplotlib.pyplot as plt

def uniform_noise(n_steps, n):
    return torch.randint(0, n_steps, (n,))

def extreme_shifted_noise(n_steps, n):
    x = random.random()
    if x < 0.25:
        return torch.randint(0, int(n_steps / 10), (n,))
    elif x > 0.75:
        return torch.randint(int(n_steps / 10) * 9, n_steps, (n,))
    return torch.randint(0, n_steps, (n,))

def u_dist_abs_noise(n_steps, n):
    sampling_probs = []
    for i in range(n_steps):
        sampling_probs.append(abs(i - (n_steps / 2)) + + n_steps / 8)

    sampling_probs = torch.tensor([i / sum(sampling_probs) for i in sampling_probs])
    return torch.multinomial(sampling_probs, n, replacement=True)

def u_dist_quadratic_noise(n_steps, n):
    sampling_probs = []
    for i in range(n_steps):
        sampling_probs.append((i - (n_steps / 2)) ** 2 + (n_steps / 8) ** 2)

    sampling_probs = torch.tensor([i / sum(sampling_probs) for i in sampling_probs])
    return torch.multinomial(sampling_probs, n, replacement=True)

def u_dist_quartic_noise(n_steps, n):
    sampling_probs = []
    for i in range(n_steps):
        sampling_probs.append((i - (n_steps / 2)) ** 4 + (n_steps / 8) ** 4)
    sampling_probs = torch.tensor([i / sum(sampling_probs) for i in sampling_probs])
    return torch.multinomial(sampling_probs, n, replacement=True)

def training_loop(ddpm, loader, n_epochs, optim, device, display=False, store_path="ddpm_model.pt", noise_dist=uniform_noise, noise_dist_label_str='uniform'):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    epoch_losses = []

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            # Loading data
            x0 = batch[0].to(device)
            n = len(x0)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            t = noise_dist(n_steps, n).to(device)

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0, t, eta)

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        # Display images generated at this epoch
        if display:
            show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"
        epoch_losses.append(epoch_loss)

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)
    
    plt.bar(np.arange(len(epoch_losses)), epoch_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss per Epoch Noise Dist: {noise_dist_label_str.title()} ')
    plt.savefig(f'loss_{noise_dist_label_str}.png', format='png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_dist', type=str, default='uniform')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors
    no_train = False
    batch_size = 128
    n_epochs = 20
    lr = 0.001

    noise_dist_functions = {
        'uniform': uniform_noise,
        'extreme_shifted': extreme_shifted_noise,
        'u_dist_abs': u_dist_abs_noise,
        'u_dist_quadratic': u_dist_quadratic_noise,
        'u_dist_quartic': u_dist_quartic_noise
    }
    noise_sampling_dist = noise_dist_functions[args.noise_dist]
    noise_sampling_dist_name = args.noise_dist
    store_path = f"ddpm_model_mnist_{noise_sampling_dist_name}.pt"

    transform = Compose([
        ToTensor(),
        Lambda(lambda x: (x - 0.5) * 2)]
    )
    ds_fn = MNIST
    dataset = ds_fn("./datasets", download=True, train=True, transform=transform)
    loader = DataLoader(dataset, batch_size, shuffle=True)

    ddpm = DDPM(UNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)
    training_loop(ddpm, loader, n_epochs, optim=Adam(ddpm.parameters(), lr), device=device, store_path=store_path, noise_dist=noise_sampling_dist, noise_dist_label_str=noise_sampling_dist_name)