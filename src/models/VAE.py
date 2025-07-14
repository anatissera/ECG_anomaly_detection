import numpy as np
import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    def __init__(self, ch, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size//2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size, padding=padding, dilation=dilation), nn.LeakyReLU(),
            nn.Conv1d(ch, ch, kernel_size, padding=padding, dilation=dilation)
        )
    def forward(self, x):
        return x + self.block(x)

torch.manual_seed(0)
class VAE1D(nn.Module):
    def __init__(self, input_ch=1, coeff_ch=55, latent_dim=16, seq_len=2048, n_blocks=3):
        super().__init__()
        
        # Encoder
        layers = [nn.Conv1d(input_ch, 32, 19, 2, 9), nn.LeakyReLU()]
        for i in range(n_blocks):
            layers.append(ResBlock1D(32, 3, 2**i))
        self.signal_encoder = nn.Sequential(*layers)
        
        self.out_len = seq_len // 2
        flat_signal_size = 32 * self.out_len
        
        self.flatten = nn.Flatten()
        
        # Dense layers 
        # (flattened ECG features + FMM coefficients)
        self.fc_mu = nn.Linear(flat_signal_size + coeff_ch, latent_dim)
        self.fc_logv = nn.Linear(flat_signal_size + coeff_ch, latent_dim)
        
        # Decoder (reconstruye solo la señal ECG)
        self.fc_dec = nn.Linear(latent_dim, flat_signal_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32, self.out_len)),
            nn.ConvTranspose1d(32, input_ch, 19, 2, 9, output_padding=1)
        )

    def encode(self, x_signal, x_coeffs):
        h_signal = self.signal_encoder(x_signal)
        h_signal_flat = self.flatten(h_signal)
        
        h_combined = torch.cat([h_signal_flat, x_coeffs], dim=1)
        
        # Proyectar las features combinadas al espacio latente
        return self.fc_mu(h_combined), self.fc_logv(h_combined)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h_decoded = self.fc_dec(z)
        return self.decoder(h_decoded)

    def forward(self, x_signal, x_coeffs):
        mu, logv = self.encode(x_signal, x_coeffs)
        z = self.reparameterize(mu, logv)
        recon = self.decode(z)
        return recon, mu, logv
    

def compute_scores(model, loader, device, beta):
    model.eval()
    errs, zs = [], []
    with torch.no_grad():
        for x_signal, x_coeffs, _ in loader:
            x_signal, x_coeffs = x_signal.to(device), x_coeffs.to(device)
            
            mu, logv = model.encode(x_signal, x_coeffs)
            z = model.reparameterize(mu, logv)
            rec = model.decode(z)
            
            mse = ((rec - x_signal)**2).mean(dim=[1, 2]).cpu().numpy()
            kl = (-0.5 * (1 + logv - mu.pow(2) - logv.exp()).sum(dim=1)).cpu().numpy()
            
            elbo = -mse - beta * kl
            
            errs.append(elbo)
            zs.append(mu.cpu().numpy())
            
    return np.concatenate(errs), np.vstack(zs)