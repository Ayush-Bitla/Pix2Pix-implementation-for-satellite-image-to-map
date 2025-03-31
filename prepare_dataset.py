import os
import shutil
from PIL import Image
import argparse
from tqdm import tqdm

def split_and_save_images(source_dir, target_dir):
    """
    Split images from the maps dataset where each image has a satellite view (left) 
    and a map view (right) side by side.
    
    Args:
        source_dir (str): Directory containing the original combined images
        target_dir (str): Base directory where to save the separated images
    """
    # Create target directories if they don't exist
    satellite_dir = os.path.join(target_dir, 'satellite')
    map_dir = os.path.join(target_dir, 'map')
    
    os.makedirs(satellite_dir, exist_ok=True)
    os.makedirs(map_dir, exist_ok=True)
    
    # Get all image files from source directory
    image_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Found {len(image_files)} images in {source_dir}")
    
    # Process each image
    for img_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(source_dir, img_name)
        
        try:
            # Open the image
            img = Image.open(img_path)
            
            # Get image width and divide by 2 to separate satellite and map views
            width, height = img.size
            mid_point = width // 2
            
            # Split the image
            satellite_img = img.crop((0, 0, mid_point, height))
            map_img = img.crop((mid_point, 0, width, height))
            
            # Save the split images
            satellite_img.save(os.path.join(satellite_dir, img_name))
            map_img.save(os.path.join(map_dir, img_name))
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for Pix2Pix training by splitting combined images')
    parser.add_argument('--source_train', type=str, required=True, help='Directory containing training images')
    parser.add_argument('--source_val', type=str, required=True, help='Directory containing validation images')
    parser.add_argument('--target_dir', type=str, default='data', help='Target directory for processed images')
    
    args = parser.parse_args()
    
    # Process training images
    print("Processing training images...")
    train_target = os.path.join(args.target_dir, 'train')
    os.makedirs(train_target, exist_ok=True)
    split_and_save_images(args.source_train, train_target)
    
    # Process validation images
    print("Processing validation images...")
    val_target = os.path.join(args.target_dir, 'val')
    os.makedirs(val_target, exist_ok=True)
    split_and_save_images(args.source_val, val_target)
    
    print("Dataset preparation completed!")

if __name__ == "__main__":
    main() 