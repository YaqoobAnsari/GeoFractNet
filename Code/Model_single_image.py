import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from torchsummary import summary

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


def load_model(model_path, device):
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_image(model, image, device):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    prediction = (output > 0.5).float().cpu().squeeze().numpy()
    return prediction


def patch_image(image_path, patch_size=224):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    patches = []

    for i in range(0, width, patch_size):
        for j in range(0, height, patch_size):
            box = (i, j, i + patch_size, j + patch_size)
            patch = image.crop(box)
            patches.append((patch, box))

    return patches, (width, height)


def save_patches(patches, save_dir, base_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    patch_paths = []

    for idx, (patch, box) in enumerate(patches):
        patch_path = os.path.join(save_dir, f"{base_name}_patch_{idx}.png")
        patch.save(patch_path)
        patch_paths.append((patch_path, box))

    return patch_paths


# Convert non-PNG images to PNG and remove original files
def convert_images_to_png(folder_path):
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            with Image.open(file_path) as img:
                png_filename = f"{os.path.splitext(filename)[0]}.png"
                png_path = os.path.join(folder_path, png_filename)
                img.save(png_path)
            os.remove(file_path)
            print(f"Converted {filename} to {png_filename} and removed the original file.")



def save_prediction(prediction, save_path):
    plt.imsave(save_path, prediction, cmap='gray')


def stitch_patches(patches, image_size, patch_size=224):
    width, height = image_size
    mask = np.zeros((height, width))

    for patch, box in patches:
        i, j, i_end, j_end = box

        patch_width = min(patch_size, width - i)
        patch_height = min(patch_size, height - j)

        mask[j:j + patch_height, i:i + patch_width] = patch[:patch_height, :patch_width]

    return mask



if __name__ == "__main__":
    model_path = 'C:/Users/User/Desktop/Edges/best_unet_model_17_15_24.pth'
    test_image_dir = 'C:/Users/User/Desktop/Edges/Test Images'
    patched_image_dir = 'C:/Users/User/Desktop/Edges/test_image_patched'
    merged_mask_dir = 'C:/Users/User/Desktop/Edges/Merged Masks'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Converting images to png")
    convert_images_to_png(test_image_dir)
    print("All images converted to png")

    model = load_model(model_path, device)

    if not os.path.exists(merged_mask_dir):
        os.makedirs(merged_mask_dir)

    for image_name in os.listdir(test_image_dir):
        image_path = os.path.join(test_image_dir, image_name)
        base_name, _ = os.path.splitext(image_name)

        # Patch the large image
        patches, image_size = patch_image(image_path)
        patch_paths = save_patches(patches, patched_image_dir, base_name)

        # Run the model on each patch and save predictions
        predictions = []
        for patch_path, box in patch_paths:
            patch_image_obj = Image.open(patch_path)
            prediction = predict_image(model, patch_image_obj, device)
            prediction_path = os.path.join(patched_image_dir, f"{base_name}_patch_{box[0]}_{box[1]}_prediction.png")
            save_prediction(prediction, prediction_path)
            predictions.append((prediction, box))

        # Stitch the patches back into a single large mask
        merged_mask = stitch_patches(predictions, image_size)
        merged_mask_path = os.path.join(merged_mask_dir, f"{base_name}_merged_mask.png")
        plt.imsave(merged_mask_path, merged_mask, cmap='gray')

    print("Processing completed and results saved.")
