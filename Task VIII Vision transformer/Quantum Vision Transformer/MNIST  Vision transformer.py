import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# 1. Download and load the MNIST dataset
# -------------------------------
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -------------------------------
# 2. Define the Vision Transformer model
# -------------------------------
class ViT(nn.Module):
    def __init__(self, image_size=28, patch_size=7, num_classes=10,
                 dim=64, depth=2, heads=4, mlp_dim=128, dropout=0.1):
        """
        image_size: size of the input image (28 for MNIST)
        patch_size: size of each patch (e.g., 7x7)
        num_classes: number of classification classes (10 for MNIST)
        dim: dimension of the patch embedding
        depth: number of Transformer encoder layers
        heads: number of heads in multi-head attention
        mlp_dim: hidden layer size in the feed-forward network
        dropout: dropout rate
        """
        super().__init__()
        
        # Divide the image into patches and perform a linear projection via a convolution
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        # Calculate the number of patches: (image_size/patch_size)^2
        self.num_patches = (image_size // patch_size) ** 2
        
        # Create a learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # Create learnable position embeddings (for both class token and patch tokens)
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.num_patches, dim))
        self.dropout = nn.Dropout(dropout)
        
        # Define the Transformer encoder using PyTorch's built-in TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
                                                   dim_feedforward=mlp_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Apply layer normalization
        self.norm = nn.LayerNorm(dim)
        # Classification head that outputs logits for each class
        self.head = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        # x shape: (batch, 1, 28, 28)
        x = self.patch_embed(x)             # (batch, dim, 4, 4) if patch_size=7
        x = x.flatten(2)                    # (batch, dim, num_patches)
        x = x.transpose(1, 2)               # (batch, num_patches, dim)
        
        batch_size = x.shape[0]
        # Expand the class token to match the batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)   # (batch, 1+num_patches, dim)
        x = x + self.pos_embed                # Add positional encoding
        x = self.dropout(x)
        
        # Transformer requires (sequence_length, batch, dim)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        
        # Extract the class token output
        x = self.norm(x[:, 0])
        out = self.head(x)
        return out

# -------------------------------
# 3. Training and Testing Functions
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# -------------------------------
# 4. Run Training and Testing
# -------------------------------
num_epochs = 20
train_losses = []

for epoch in range(num_epochs):
    loss = train(model, train_loader, criterion, optimizer, device)
    train_losses.append(loss)
    test_acc = test(model, test_loader, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Test Acc: {test_acc*100:.2f}%")

# Plot and save the training loss curve
plt.plot(train_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Vision Transformer on MNIST")
plt.savefig("vit_mnist_training_loss.png")  # Save as PNG in the current folder
plt.show()
