import torch
import torchvision
from utils.prior_models import UnetScoreNetwork_SR
from loss_calc import denoising_score_matching
import os
import numpy as np
from PIL import Image


class FilteredCroppedMNIST(torch.utils.data.Dataset):
    def __init__(self):

        self.mnist_data = torchvision.datasets.MNIST("mnist", download=True)

        excluded_indices = []
        for digit in range(10):
            first_idx = (self.mnist_data.targets == digit).nonzero(as_tuple=True)[0][0].item()
            excluded_indices.append(first_idx)

        all_indices = torch.arange(len(self.mnist_data.targets))
        keep_indices = all_indices[~torch.isin(all_indices, torch.tensor(excluded_indices))]

        # Filter images
        self.images = self.mnist_data.data[keep_indices]
        self.labels = self.mnist_data.targets[keep_indices]

        # Define transformations (resize after custom cropping)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]  # Get raw image (28x28)
        label = self.labels[idx]

        image = image.detach().numpy()

        valid_rows = np.any(image != 0, axis=1)
        row_indices = np.where(valid_rows)[0]
        valid_cols = np.any(image != 0, axis=0)
        col_indices = np.where(valid_cols)[0]

        image_corp = image[row_indices[0]: row_indices[-1] + 1, col_indices[0]: col_indices[-1] + 1]

        image_corp = Image.fromarray(image_corp).resize((28, 28), Image.Resampling.LANCZOS)

        image_corp = self.transform(image_corp)

        return image_corp, label


# Consts
lr = 3e-4
batch_size = 64
epochs = 400
checkpoint_path = "../checkpoints/mnist_prior_28X128.pth"

print("Initialize network")
# Initialize network
score_network = UnetScoreNetwork_SR()
print("Initialize optimizer")
# Initialize optimizer
optimizer = torch.optim.Adam(score_network.parameters(), lr=lr)
print("Initialize data")
# Initialize data
filtered_cropped_dataset = FilteredCroppedMNIST()

data_loader = torch.utils.data.DataLoader(filtered_cropped_dataset, batch_size=batch_size, shuffle=True)

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

score_network = score_network.to(device)

print("Start training")
# Training loop
for i_epoch in range(epochs):
    total_loss = 0
    for batch, _ in data_loader:
        batch = batch.reshape(batch.shape[0], -1).to(device)
        optimizer.zero_grad()
        loss = denoising_score_matching(score_network, batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * batch.shape[0]
    if i_epoch % 50 == 0:
        print(total_loss)
print("Finished training")
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
torch.save(score_network.state_dict(), checkpoint_path)
