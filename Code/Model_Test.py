# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:43:49 2024

@author: User
"""

import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
from datetime import datetime
import matplotlib.pyplot as plt


class PatchDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        original_image_path = self.data.iloc[idx, 0]
        binary_mask_path = self.data.iloc[idx, 1]

        original_image = Image.open(original_image_path).convert("RGB")
        binary_mask = Image.open(binary_mask_path).convert("L")  # Convert to grayscale

        if self.transform:
            original_image = self.transform(original_image)
            binary_mask = self.transform(binary_mask)

        binary_mask = (binary_mask > 0).float()  # Ensure binary mask is in range [0, 1]

        return original_image, binary_mask

def get_dataloaders(train_csv, val_csv, test_csv, batch_size=32, num_workers=4):
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create datasets
    train_dataset = PatchDataset(csv_file=train_csv, transform=transform)
    val_dataset = PatchDataset(csv_file=val_csv, transform=transform)
    test_dataset = PatchDataset(csv_file=test_csv, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = CBR(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = CBR(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.encoder4 = CBR(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = CBR(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = CBR(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = CBR(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = CBR(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = CBR(128, 64)

        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv_last(dec1))

 

def load_model(model_path):
    print("Loading model...")
    
    model = UNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    model.to(device)
  
    # Evaluation of the best saved model using the test dataset
    print("Evaluating the best model on the test dataset...")

    # Load the best saved model without DataParallel
    model = UNet()
    model.load_state_dict(torch.load('best_unet_model.pth', map_location=device))
    model.to(device)
    return model

def process_image(image_path, patch_size=224):
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    width, height = image.size

    # Prepare patches
    patches = []
    coordinates = []
    
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            if i + patch_size <= height and j + patch_size <= width:
                patch = image.crop((j, i, j + patch_size, i + patch_size))
                patches.append(patch)
                coordinates.append((i, j))
    
    print(f"Image size: {width}x{height}, Patches prepared: {len(patches)}")
    return image, patches, coordinates

def predict_patches(model, patches, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    predictions = []
    print("Predicting patches...")
    with torch.no_grad():
        for idx, patch in enumerate(patches):
            input_tensor = transform(patch).unsqueeze(0).to(device)
            output = model(input_tensor)
            prediction = torch.sigmoid(output).cpu().numpy()[0, 0]
            predictions.append(prediction)
            print(f"Patch {idx+1}/{len(patches)} predicted.")
    
    print("All patches predicted.")
    return predictions

def stitch_patches(predictions, coordinates, image_size, patch_size=224):
    height, width = image_size
    stitched_mask = np.zeros((height, width), dtype=np.uint8)

    for prediction, (i, j) in zip(predictions, coordinates):
        patch_height = min(patch_size, height - i)
        patch_width = min(patch_size, width - j)
        
        if patch_height > 0 and patch_width > 0:
            stitched_mask[i:i+patch_height, j:j+patch_width] = (prediction[:patch_height, :patch_width] > 0.5).astype(np.uint8) * 255
    
    print("Patches stitched into final mask.")
    return stitched_mask


def save_mask(mask, save_path):
    mask_image = Image.fromarray(mask)
    mask_image.save(save_path)
    print(f"Mask saved: {save_path}")

def main(test_images_folder, model_path, output_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path).to(device)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    for image_name in os.listdir(test_images_folder):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_images_folder, image_name)
            image, patches, coordinates = process_image(image_path)
            predictions = predict_patches(model, patches, device)
            mask = stitch_patches(predictions, coordinates, image.size)
            save_path = os.path.join(output_folder, f"mask_{image_name}")
            save_mask(mask, save_path)
            print(f"Processed and saved mask for {image_name}\n")

# Example usage
test_images_folder = "Test Image"
model_path = r"G:/Edges/best_unet_model.pth"
output_folder = "Test Output Masks"

main(test_images_folder, model_path, output_folder)
