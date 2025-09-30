import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import csv
from datetime import datetime


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

def get_dataloaders(train_csv, val_csv, test_csv, batch_size=64, num_workers=4):
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


class SobelEdgeDetectionLayer(nn.Module):
    def __init__(self, in_channels):
        super(SobelEdgeDetectionLayer, self).__init__()
        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        
        sobel_kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).repeat(in_channels, 1, 1, 1)
        sobel_kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).repeat(in_channels, 1, 1, 1)
        
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)
    
    def forward(self, x):
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edge_features = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edge_features

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels, dropout_prob=0.15):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob)
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

        self.edge_det1 = SobelEdgeDetectionLayer(64)
        self.edge_det2 = SobelEdgeDetectionLayer(128)
        self.edge_det3 = SobelEdgeDetectionLayer(256)
        self.edge_det4 = SobelEdgeDetectionLayer(512)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        enc4_edge = self.edge_det4(enc4)
        dec4 = torch.cat((dec4, enc4_edge), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        enc3_edge = self.edge_det3(enc3)
        dec3 = torch.cat((dec3, enc3_edge), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        enc2_edge = self.edge_det2(enc2)
        dec2 = torch.cat((dec2, enc2_edge), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        enc1_edge = self.edge_det1(enc1)
        dec1 = torch.cat((dec1, enc1_edge), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv_last(dec1))


def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def lr_schedule(epoch):
    if epoch < 10:
        return 1.0
    elif epoch < 20:
        return 0.5
    elif epoch < 30:
        return 0.1
    else:
        return 0.05

'''
class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)  # Ensure inputs are in the range [0, 1]
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        bce = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        intersection = (inputs * targets).sum()
        dice = 1 - (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        return bce + dice
'''

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.clamp(min=0, max=1)  # Ensure preds are in range [0, 1]
        bce_loss = self.bce(preds, targets)
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (preds * targets).sum()
        dice_loss = 1 - ((2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth))
        return bce_loss + dice_loss

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pth')
        self.val_loss_min = val_loss


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def check_for_nans(tensor, name):
    if tensor is None:
        print(f"NoneType found in {name}")
    else:
        if torch.isnan(tensor).any():
            print(f"NaN values found in {name}")
        if torch.isinf(tensor).any():
            print(f"Infinite values found in {name}")

def reinitialize_layer(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.constant_(layer.weight, 1)
        nn.init.constant_(layer.bias, 0)

def check_and_reinitialize(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"NaN or Inf detected in gradients of {name}. Reinitializing layer.")
                layer_name = name.split('.')[0]
                layer = getattr(model, layer_name)
                reinitialize_layer(layer)


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
    model.apply(weights_init)

    lr = 0.002
    num_epochs = 125
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

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
    print(f"Training model using device {device}")

    model.to(device)

    # Early stopping
    early_stopping = EarlyStopping(patience=5, verbose=True)

    # Training loop
    print("Starting training...")
    print(f"Training Info: lr = {lr}, Epochs = {num_epochs}, Model = UNet, Optimizer = {optimizer}")
    print(f"{'Epoch':<5} {'Time':<10} {'Train Loss':<12} {'Val Loss':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'IoU':<10} {'Lr':<12}")
    print("="*100)
    
    # Add this to initialize the CSV file
    model_info = f"UNet1_lr{lr}_epochs{num_epochs}"
    timestamp = datetime.now().strftime("%H_%M_%S")
    csv_filename = f"training_log_{model_info}_{timestamp}.csv"
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Time", "Train Loss", "Val Loss", "Accuracy", "Precision", "Recall", "F1 Score", "IoU", "Learning Rate", "Loss Function", "Optimizer"])

    best_val_loss = float('inf')
    max_norm = 1.0  # Set an appropriate value for gradient clipping

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        start_time = time.time()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            if torch.isnan(images).any() or torch.isnan(masks).any():
                print("NaN values found in inputs")
            if torch.isinf(images).any() or torch.isinf(masks).any():
                print("Infinite values found in inputs")
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.clamp(min=0, max=1) 
            check_for_nans(outputs, "outputs before loss")
            if torch.isnan(outputs).any():
                print("NaN values found in outputs")
            if torch.isinf(outputs).any():
                print("Infinite values found in outputs")
            loss = criterion(outputs, masks)
            check_for_nans(loss, "loss")
            loss.backward()
            for name, param in model.named_parameters():
                check_for_nans(param.grad, f"gradient of {name}")
                if param.grad is not None and (param.grad.abs() > 1e2).any():
                    print(f"Large gradient values found in {name}")
            
            check_and_reinitialize(model)
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            running_loss += loss.item() * images.size(0)
    
        epoch_loss = running_loss / len(train_loader.dataset)
    
        # Validation step
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                outputs = outputs.clamp(min=0, max=1)  # Ensure outputs are in range [0, 1]
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
    
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(masks.cpu().numpy().flatten())
    
        val_loss = val_loss / len(val_loader.dataset)
    
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        iou = jaccard_score(all_labels, all_preds)
    
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{epoch+1:<5} {epoch_time:<10.2f} {epoch_loss:<12.6f} {val_loss:<12.6f} {accuracy:<10.6f} {precision:<10.6f} {recall:<10.6f} {f1:<10.6f} {iou:<10.6f} {current_lr:<12.6f}")
    
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, epoch_time, epoch_loss, val_loss, accuracy, precision, recall, f1, iou, current_lr, "BCEDiceLoss", "Adam"])

        # Check early stopping criteria
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), f'best_unet_model_{timestamp}.pth')
            print("Best model saved.")

        # Update the learning rate
        scheduler.step()