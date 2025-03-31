import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# Check CUDA availability
def check_cuda():
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available! Training will run on CPU which is significantly slower.")
        print("If you have an NVIDIA GPU, please make sure you have installed the proper CUDA drivers and PyTorch with CUDA support.")
        return False
    else:
        # Set CUDA device to 0 (first GPU)
        torch.cuda.set_device(0)
        # Enable cuDNN benchmark for faster training
        torch.backends.cudnn.benchmark = True
        print("CUDA enabled successfully!")
        return True

# Print CUDA information to verify GPU usage
def print_cuda_info():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        # Print initial GPU memory usage
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print("cuDNN benchmark enabled for faster training")

# Lighter UNet Generator for faster training
class LightUNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(LightUNetGenerator, self).__init__()
        
        # Initial downsampling layer (no BatchNorm)
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        # Downsampling layers
        self.down2 = self._downsample_block(features, features * 2)
        self.down3 = self._downsample_block(features * 2, features * 4)
        self.down4 = self._downsample_block(features * 4, features * 8)
        self.down5 = self._downsample_block(features * 8, features * 8)
        self.down6 = self._downsample_block(features * 8, features * 8)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, stride=2, padding=1),
            nn.ReLU(True)
        )
        
        # Upsampling layers
        self.up1 = self._upsample_block(features * 8, features * 8, dropout=True)
        self.up2 = self._upsample_block(features * 8 * 2, features * 8, dropout=True)
        self.up3 = self._upsample_block(features * 8 * 2, features * 8)
        self.up4 = self._upsample_block(features * 8 * 2, features * 4)
        self.up5 = self._upsample_block(features * 4 * 2, features * 2)
        self.up6 = self._upsample_block(features * 2 * 2, features)
        
        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def _downsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )
    
    def _upsample_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        bottleneck = self.bottleneck(d6)
        
        # Decoder with skip connections
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d6], dim=1))
        up3 = self.up3(torch.cat([up2, d5], dim=1))
        up4 = self.up4(torch.cat([up3, d4], dim=1))
        up5 = self.up5(torch.cat([up4, d3], dim=1))
        up6 = self.up6(torch.cat([up5, d2], dim=1))
        
        return self.final(torch.cat([up6, d1], dim=1))

# Lighter PatchGAN Discriminator
class LightPatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6, features=64):
        super(LightPatchGANDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Layer 1 (no BatchNorm)
            nn.Conv2d(in_channels, features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            
            # Layer 2
            nn.Conv2d(features, features * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, True),
            
            # Layer 3
            nn.Conv2d(features * 2, features * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, True),
            
            # Layer 4 (output layer)
            nn.Conv2d(features * 4, 1, 4, stride=1, padding=1),
            nn.Sigmoid()  # Using Sigmoid directly to avoid separate BCEWithLogitsLoss for speed
        )
        
    def forward(self, x, y):
        # Concatenate input and output channels
        return self.model(torch.cat([x, y], dim=1))

# Custom dataset class for satellite-map image pairs
class SatelliteMapDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        # Paths to satellite and map images
        self.satellite_dir = os.path.join(root_dir, mode, 'satellite')
        self.map_dir = os.path.join(root_dir, mode, 'map')
        
        # Check if directories exist
        if not os.path.exists(self.satellite_dir) or not os.path.exists(self.map_dir):
            raise ValueError(f"Directories not found: {self.satellite_dir} or {self.map_dir}")
        
        # Get image filenames (assuming both directories have the same files)
        self.image_files = [f for f in os.listdir(self.satellite_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        satellite_path = os.path.join(self.satellite_dir, img_name)
        map_path = os.path.join(self.map_dir, img_name)
        
        # Load images
        satellite_img = Image.open(satellite_path).convert('RGB')
        map_img = Image.open(map_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            satellite_img = self.transform(satellite_img)
            map_img = self.transform(map_img)
            
        return satellite_img, map_img

# Initialize weights function - for faster convergence
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Calculate discriminator output size for debugging purposes
def calculate_discriminator_output_size():
    # Create dummy discriminator
    disc = LightPatchGANDiscriminator()
    # Create dummy inputs (batch_size, channels, height, width)
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)
    # Get output
    output = disc(x, y)
    # Return output shape
    return output.shape

# Pix2Pix model training function with optimization for speed
def train_pix2pix(generator, discriminator, train_dataloader, val_dataloader, num_epochs, device, 
                  save_dir='./checkpoints', sample_dir='./samples', save_interval=5):
    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Loss functions - use BCE loss directly since we use Sigmoid in the discriminator
    criterion_gan = nn.BCELoss()
    criterion_l1 = nn.L1Loss()
    
    # Optimizers with slightly higher learning rate for faster training
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0003, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.5, 0.999))
    
    # To track best validation loss
    best_val_loss = float('inf')
    
    # For timing training
    from datetime import datetime
    start_time = datetime.now()
    
    # Calculate the actual discriminator output size for a 256x256 input image
    # Since we can't use calculate_discriminator_output_size directly here due to device mismatch,
    # we'll hardcode the value (31,31) based on our architecture
    patch_size = 31  # The actual output size from our discriminator
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        generator.train()
        discriminator.train()
        
        # Track losses
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        for satellite, real_map in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            # Move data to the selected device
            satellite = satellite.to(device, non_blocking=True)  # non_blocking for async transfer
            real_map = real_map.to(device, non_blocking=True)
            batch_size = satellite.size(0)
            
            # Ground truths - make sure these are on the device
            real_label = torch.ones(batch_size, 1, patch_size, patch_size, device=device)
            fake_label = torch.zeros(batch_size, 1, patch_size, patch_size, device=device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_d.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            # Generate fake maps
            with torch.no_grad():
                fake_map = generator(satellite)
                
            # Real pairs
            real_output = discriminator(satellite, real_map)
            d_real_loss = criterion_gan(real_output, real_label)
            
            # Fake pairs
            fake_output = discriminator(satellite, fake_map.detach())
            d_fake_loss = criterion_gan(fake_output, fake_label)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            
            d_loss.backward()
            optimizer_d.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_g.zero_grad(set_to_none=True)
            
            # Generate fake maps again (for generator training)
            fake_map = generator(satellite)
            
            # GAN loss
            fake_output = discriminator(satellite, fake_map)
            g_gan_loss = criterion_gan(fake_output, real_label)
            
            # L1 loss (pixel-wise)
            g_l1_loss = criterion_l1(fake_map, real_map) * 100  # lambda=100
            
            # Total generator loss
            g_loss = g_gan_loss + g_l1_loss
            
            g_loss.backward()
            optimizer_g.step()
            
            # Track losses
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
        
        # Print epoch losses
        epoch_d_loss /= len(train_dataloader)
        epoch_g_loss /= len(train_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], D Loss: {epoch_d_loss:.4f}, G Loss: {epoch_g_loss:.4f}')
        
        # Validation phase - but only every few epochs to save time
        if epoch % 2 == 0:
            generator.eval()
            val_g_loss = 0.0
            
            with torch.no_grad():
                for i, (satellite, real_map) in enumerate(val_dataloader):
                    if i >= len(val_dataloader) // 4:  # Only process 25% of validation data for speed
                        break
                        
                    satellite = satellite.to(device, non_blocking=True)
                    real_map = real_map.to(device, non_blocking=True)
                    
                    # Generate fake maps
                    fake_map = generator(satellite)
                    
                    # L1 loss for validation
                    val_loss = criterion_l1(fake_map, real_map) * 100
                    val_g_loss += val_loss.item()
                    
                    # Save sample images
                    if i == 0 and epoch % save_interval == 0:
                        # Save a few samples from the validation set
                        sample_count = min(3, satellite.size(0))
                        fig, axs = plt.subplots(sample_count, 3, figsize=(12, 4*sample_count))
                        
                        for j in range(sample_count):
                            # Convert tensors to numpy arrays for visualization
                            img_satellite = satellite[j].cpu()
                            img_real = real_map[j].cpu()
                            img_fake = fake_map[j].cpu()
                            
                            # Denormalize images
                            img_satellite = (img_satellite * 0.5 + 0.5).numpy().transpose(1, 2, 0)
                            img_real = (img_real * 0.5 + 0.5).numpy().transpose(1, 2, 0)
                            img_fake = (img_fake * 0.5 + 0.5).numpy().transpose(1, 2, 0)
                            
                            # Plot images
                            axs[j, 0].imshow(img_satellite)
                            axs[j, 0].set_title('Satellite')
                            axs[j, 0].axis('off')
                            
                            axs[j, 1].imshow(img_real)
                            axs[j, 1].set_title('Real Map')
                            axs[j, 1].axis('off')
                            
                            axs[j, 2].imshow(img_fake)
                            axs[j, 2].set_title('Generated Map')
                            axs[j, 2].axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(sample_dir, f'epoch_{epoch+1}.png'))
                        plt.close()
            
            # Process validation results
            if len(val_dataloader) > 0:
                val_g_loss /= min(len(val_dataloader) // 4, len(val_dataloader))
                print(f'Validation Loss: {val_g_loss:.4f}')
                
                # Save model checkpoint
                if (epoch+1) % save_interval == 0:
                    torch.save({
                        'epoch': epoch,
                        'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                        'optimizer_g_state_dict': optimizer_g.state_dict(),
                        'optimizer_d_state_dict': optimizer_d.state_dict(),
                        'val_loss': val_g_loss,
                    }, os.path.join(save_dir, f'pix2pix_epoch_{epoch+1}.pth'))
                
                # Save best model
                if val_g_loss < best_val_loss:
                    best_val_loss = val_g_loss
                    torch.save({
                        'epoch': epoch,
                        'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                        'optimizer_g_state_dict': optimizer_g.state_dict(),
                        'optimizer_d_state_dict': optimizer_d.state_dict(),
                        'val_loss': val_g_loss,
                    }, os.path.join(save_dir, 'pix2pix_best.pth'))
                    print(f"Saved best model with validation loss: {val_g_loss:.4f}")
        
        # Print GPU memory usage
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            torch.cuda.empty_cache()
    
    # Measure total training time
    training_time = datetime.now() - start_time
    print(f"Total training time: {training_time}")
    
    # Save the final model
    torch.save(generator.state_dict(), 'saved_model_10epochs.pth')
    print("Saved final model as 'saved_model_10epochs.pth'")

def main():
    # Check CUDA availability
    check_cuda()
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Calculate and print the discriminator output size for debugging
    print(f"Discriminator output shape: {calculate_discriminator_output_size()}")
    
    # Set device - force CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Print CUDA information
    print_cuda_info()
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create datasets
    try:
        train_dataset = SatelliteMapDataset('data', mode='train', transform=transform)
        val_dataset = SatelliteMapDataset('data', mode='val', transform=transform)
        
        # Set the batch size based on available GPU memory and CPU cores
        batch_size = 16 if torch.cuda.is_available() else 8
        num_workers = 8 if torch.cuda.is_available() else 4
        
        # Create dataloaders with optimization for speed
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),  # Pin memory for faster GPU transfer
            persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between batches
            prefetch_factor=2 if num_workers > 0 else None  # Prefetch next batch
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if num_workers > 0 else False
        )
        
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Batch size: {batch_size}")
        print(f"Number of workers: {num_workers}")
        
        # Initialize models - use the lightweight versions
        generator = LightUNetGenerator().to(device)
        discriminator = LightPatchGANDiscriminator().to(device)
        
        # Apply weight initialization
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        
        # Training parameters - fewer epochs for quick results
        num_epochs = 10  # Use 10 epochs like in the Keras example
        
        # Print model summary (parameters count)
        generator_params = sum(p.numel() for p in generator.parameters())
        discriminator_params = sum(p.numel() for p in discriminator.parameters())
        print(f"Generator parameters: {generator_params:,}")
        print(f"Discriminator parameters: {discriminator_params:,}")
        
        # Train the model
        print("Starting training...")
        train_pix2pix(generator, discriminator, train_dataloader, val_dataloader, 
                     num_epochs, device, save_dir='./checkpoints', sample_dir='./samples')
        
        print("Training completed!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        # Check if data directories exist
        print("\nChecking data directories:")
        data_dirs = ['data', 'data/train', 'data/val', 
                    'data/train/satellite', 'data/train/map',
                    'data/val/satellite', 'data/val/map']
        
        for d in data_dirs:
            exists = os.path.exists(d)
            print(f"Directory '{d}' exists: {exists}")

if __name__ == "__main__":
    main() 