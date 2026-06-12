import cv2 as cv
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

# 1. SETUP DEVICE (GPU/MPS/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 2. DATA LOADING & AUGMENTATION
images = [] 
labels = []
size = 2500
count = 0

print("Loading images...")
for i in range(size + 500): # Buffer for missing/corrupt files
    if count >= size:
        break
    
    try:
        # Paths to the Kaggle dataset structure
        cat_path = f'./PetImages/Cat/{i}.jpg'
        dog_path = f'./PetImages/Dog/{i}.jpg'
        
        cat_im = cv.imread(cat_path)
        dog_im = cv.imread(dog_path)

        # Check if images actually exist and aren't corrupted
        if cat_im is not None and dog_im is not None:
            for img, lbl in [(cat_im, 0), (dog_im, 1)]:
                # Random Horizontal Flip (Augmentation)
                if np.random.rand() > 0.5:
                    img = cv.flip(img, 1)
                
                img = cv.resize(img, (224, 224))
                images.append(img)
                labels.append(lbl)
            count += 1
    except Exception:
        continue

# Convert to Tensors and Normalize
images = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2).float() / 255.0
labels = torch.tensor(labels).view(-1, 1).float()

data_set = TensorDataset(images, labels)
train_loader = DataLoader(data_set, batch_size=32, shuffle=True)

# 3. MODEL ARCHITECTURE
class ImageClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # 28
            
            nn.AdaptiveAvgPool2d((7, 7)) 
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = ImageClassification().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.BCELoss()

# 4. TRAINING LOOP
print("Starting training...")
for epoch in range(15):
    model.train()
    total_loss = 0
    for batch_images, batch_labels in train_loader:
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        pred = model(batch_images)
        loss = loss_fn(pred, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(train_loader):.4f}")

# 5. SAVE THE MODEL
torch.save(model.state_dict(), 'cat_dog_model.pth')
print("Model saved as cat_dog_model.pth")