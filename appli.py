import streamlit as st
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Film Layer
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Training function
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            st.write(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Streamlit app
def main():
    st.title("CNN Training on MNIST Dataset")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = st.slider("Number of Epochs", 1, 10, 1)
    if st.button("Start Training"):
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, criterion, epoch)
            # Display 5 examples of images and their classifications
            model.eval()
            examples = enumerate(train_loader)
            batch_idx, (example_data, example_targets) = next(examples)
            example_data, example_targets = example_data.to(device), example_targets.to(device)
            with torch.no_grad():
                output = model(example_data)
            _, preds = torch.max(output, 1)
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            for i in range(5):
                axes[i].imshow(example_data[i].cpu().numpy().squeeze(), cmap='gray')
                axes[i].set_title(f'Pred: {preds[i].item()}')
                axes[i].axis('off')
            st.pyplot(fig)

            # Display examples of images that are not well classified
            incorrect_indices = (preds != example_targets).nonzero(as_tuple=True)[0]
            if len(incorrect_indices) > 0:
                fig, axes = plt.subplots(1, min(5, len(incorrect_indices)), figsize=(15, 3))
                for i, idx in enumerate(incorrect_indices[:5]):
                    axes[i].imshow(example_data[idx].cpu().numpy().squeeze(), cmap='gray')
                    axes[i].set_title(f'True: {example_targets[idx].item()}, Pred: {preds[idx].item()}')
                    axes[i].axis('off')
                st.pyplot(fig)
        st.success("Training Completed")

if __name__ == "__main__":
    main()