
import torch.nn as nn
import torch



class DigitGenerator(nn.Module):
    def __init__(self, latent_dim=100, label_dim=10):
        super().__init__()
        self.label_embedding = nn.Embedding(label_dim, label_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_embedding(labels)
        x = torch.cat((noise, label_input), dim=1)
        x = self.model(x)
        return x.view(-1, 1, 28, 28)
