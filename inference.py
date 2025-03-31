import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pix2pix import LightUNetGenerator

def load_model(model_path, device):
    """Load the trained generator model"""
    model = LightUNetGenerator().to(device)
    
    try:
        # Handle both full checkpoints and state_dict
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['generator_state_dict'])
            print(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model weights")
        
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def process_image(image_path, transform=None):
    """Load and preprocess a satellite image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0), image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

def generate_map(model, image_tensor, device):
    """Generate a map from a satellite image tensor"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        generated = model(image_tensor)
    
    return generated

def save_result(original_image, generated_map, output_path):
    """Save the original satellite image and generated map side by side"""
    # Convert the generated map tensor to a numpy array
    generated_np = generated_map.squeeze(0).cpu().detach().numpy()
    # Denormalize from [-1, 1] to [0, 1]
    generated_np = (generated_np * 0.5 + 0.5).transpose(1, 2, 0)
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original satellite image
    ax1.imshow(original_image)
    ax1.set_title('Original Satellite Image', fontsize=14)
    ax1.axis('off')
    
    # Plot generated map
    ax2.imshow(generated_np)
    ax2.set_title('Generated Map', fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Result saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate a map from a satellite image using Pix2Pix')
    parser.add_argument('--image', required=True, help='Path to the satellite image')
    parser.add_argument('--model', default='saved_model_10epochs.pth', help='Path to the trained model')
    parser.add_argument('--output', default='result.png', help='Path to save the result')
    args = parser.parse_args()
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, device)
    if model is None:
        return
    
    # Process input image
    image_tensor, original_image = process_image(args.image)
    if image_tensor is None:
        return
    
    # Generate map
    print("Generating map...")
    generated_map = generate_map(model, image_tensor, device)
    
    # Save result
    save_result(original_image, generated_map, args.output)

if __name__ == "__main__":
    main() 