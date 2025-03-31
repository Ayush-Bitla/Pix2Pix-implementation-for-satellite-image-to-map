import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import argparse
import random
from pix2pix import LightUNetGenerator

def load_model(checkpoint_path, device):
    """Load the trained generator model"""
    # Initialize the generator
    model = LightUNetGenerator().to(device)
    
    print(f"Loading model from {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        # Load model weights - support both full checkpoints and state dicts
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['generator_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state dict")
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

def process_image(image_path, device, transform=None):
    """Load and preprocess a single image"""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor, image

def denormalize(tensor):
    """Convert normalized tensor back to image format"""
    tensor = tensor.clone().detach().cpu()
    tensor = tensor * 0.5 + 0.5  # Denormalize
    tensor = tensor.clamp(0, 1)
    return tensor.numpy().transpose(1, 2, 0)

def generate_and_save(model, input_tensor, output_path="generated_map.png"):
    """Generate a map from a satellite image and save the result"""
    with torch.no_grad():
        generated = model(input_tensor)
    
    # Convert tensor to image
    generated_np = denormalize(generated[0])
    
    # Save the generated image
    plt.figure(figsize=(8, 8))
    plt.imshow(generated_np)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Generated image saved to {output_path}")
    return generated_np

def plot_comparison(input_img, generated_img, target_img=None, save_path="comparison.png"):
    """Plot input, generated, and target images side by side"""
    if target_img is not None:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        titles = ['Satellite Image', 'Generated Map', 'Real Map']
        imgs = [input_img, generated_img, target_img]
        
        for i, (ax, img, title) in enumerate(zip(axs, imgs, titles)):
            ax.imshow(img)
            ax.set_title(title, fontsize=14)
            ax.axis('off')
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        titles = ['Satellite Image', 'Generated Map']
        imgs = [input_img, generated_img]
        
        for i, (ax, img, title) in enumerate(zip(axs, imgs, titles)):
            ax.imshow(img)
            ax.set_title(title, fontsize=14)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Test a trained Pix2Pix model')
    parser.add_argument('--model', default='saved_model_10epochs.pth', help='Path to the model checkpoint')
    parser.add_argument('--data_dir', default='data/val', help='Directory with test data')
    parser.add_argument('--output_dir', default='results', help='Directory to save results')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load the trained model
        model = load_model(args.model, device)
        
        # Setup image transformation
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Find available test images
        satellite_dir = os.path.join(args.data_dir, 'satellite')
        map_dir = os.path.join(args.data_dir, 'map')
        
        if not os.path.exists(satellite_dir):
            raise FileNotFoundError(f"Satellite images directory not found: {satellite_dir}")
        
        image_files = [f for f in os.listdir(satellite_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if not image_files:
            raise ValueError("No images found in the specified directory")
        
        print(f"Found {len(image_files)} test images")
        
        # Randomly select images to test
        num_samples = min(5, len(image_files))
        selected_files = random.sample(image_files, num_samples)
        
        for i, filename in enumerate(selected_files):
            # Process satellite image
            satellite_path = os.path.join(satellite_dir, filename)
            satellite_tensor, satellite_img = process_image(satellite_path, device, transform)
            satellite_np = denormalize(satellite_tensor[0])
            
            # Generate map from satellite image
            generated_np = generate_and_save(
                model, 
                satellite_tensor, 
                output_path=os.path.join(args.output_dir, f"generated_{i}.png")
            )
            
            # If target map exists, include it in the comparison
            map_path = os.path.join(map_dir, filename)
            if os.path.exists(map_path):
                target_tensor, target_img = process_image(map_path, device, transform)
                target_np = denormalize(target_tensor[0])
                
                # Plot and save comparison
                plot_comparison(
                    satellite_np, 
                    generated_np, 
                    target_np, 
                    save_path=os.path.join(args.output_dir, f"comparison_{i}.png")
                )
            else:
                # Plot and save comparison without target
                plot_comparison(
                    satellite_np, 
                    generated_np, 
                    save_path=os.path.join(args.output_dir, f"comparison_{i}.png")
                )
        
        print(f"Testing completed! Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main() 