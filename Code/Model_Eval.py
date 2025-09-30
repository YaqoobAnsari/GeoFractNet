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


def compute_iou(preds, masks, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * masks).sum((1, 2, 3))
    union = preds.sum((1, 2, 3)) + masks.sum((1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


if __name__ == "__main__":
    # Define paths to the CSV files
    print("Initiating Model training...")

    train_csv = 'C:/Users/User/Desktop/Edges/train.csv'
    val_csv = 'C:/Users/User/Desktop/Edges/validation.csv'
    test_csv = 'C:/Users/User/Desktop/Edges/test.csv'

    # Get the dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(train_csv, val_csv, test_csv)
    print("Data loaded...")

    # Instantiate the model, loss function, and optimizer
    model = UNet()
    lr = 0.001
    num_epochs = 10
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Model initiated, checking for available GPUs...")
    # Check for available GPUs and their cores
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"{num_gpus} GPUs available:")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} with {torch.cuda.get_device_properties(i).multi_processor_count} cores")
        model = nn.DataParallel(model)
    else:
        print("No GPUs available, using CPU")

    # Move model to the appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    model.to(device)
  
    # Evaluation of the best saved model using the test dataset
    print("Evaluating the best model on the test dataset...")

    # Load the best saved model without DataParallel
    model = UNet()
    model.load_state_dict(torch.load('C:/Users/User/Desktop/Edges/best_unet_model_17_15_24.pth', map_location=device))
    model.to(device)

    # Create directory for test results
    test_results_dir = "Model_Test_Results_17_15_24"
    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir)

    # Evaluation step
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    ious = []
    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item() * images.size(0)

            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(masks.cpu().numpy().flatten())

            # Calculate IoU for each batch and append to the list
            iou = compute_iou(outputs, masks)
            ious.append(iou)

            # Save the original and predicted masks as subplots
            for i in range(images.size(0)):
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(images[i].cpu().permute(1, 2, 0))
                axes[0].set_title("Original Image")
                axes[1].imshow(masks[i].cpu().squeeze(), cmap='gray')
                axes[1].set_title("Original Mask")
                axes[2].imshow(preds[i].cpu().squeeze(), cmap='gray')
                axes[2].set_title("Predicted Mask")
                plt.savefig(os.path.join(test_results_dir, f"result_{idx*images.size(0)+i}.png"))
                plt.close(fig)

    test_loss = test_loss / len(test_loader.dataset)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    mean_iou = sum(ious) / len(ious)

    # Save the testing metrics in a neat format
    metrics_filename = os.path.join(test_results_dir, "testing_metrics_13_20_56.txt")
    with open(metrics_filename, 'w') as f:
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Accuracy: {accuracy:.6f}\n")
        f.write(f"Precision: {precision:.6f}\n")
        f.write(f"Recall: {recall:.6f}\n")
        f.write(f"F1 Score: {f1:.6f}\n")
        f.write(f"Mean IoU: {mean_iou:.6f}\n")

    print("Testing completed and results saved.")
