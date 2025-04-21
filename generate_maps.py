#!/usr/bin/env python
# generate_maps.py
import os
import argparse
from PIL import Image
import torch
import torchvision.transforms as transforms
import sys

# Add the path to pix2pix code
sys.path.append('pytorch-CycleGAN-and-pix2pix')

from models import create_model
from options.test_options import TestOptions
from util import util


def process_image(img_path, transform):
    """Process a single image to prepare it for the model"""
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    return img.unsqueeze(0)  # Add batch dimension


def save_output_image(tensor, output_path):
    """Convert output tensor to image and save it"""
    image_numpy = util.tensor2im(tensor)
    util.save_image(image_numpy, output_path)
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate maps from satellite images using trained pix2pix model')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with input satellite images')
    parser.add_argument('--output_dir', type=str, default='./generated_maps', help='Directory to save output map images')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/sat2map_pix2pix_50epochs', 
                        help='Directory with trained model checkpoint')
    parser.add_argument('--gpu_id', type=int, default=-1, help='GPU ID (-1 for CPU)')
    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up model options
    opt = TestOptions().parse()  # get test options
    # Override some options with our custom values
    opt.dataroot = 'placeholder'  # Not used directly but needed for model creation
    opt.model = 'pix2pix'
    opt.direction = 'AtoB'
    opt.checkpoints_dir = os.path.dirname(args.checkpoint_dir)
    opt.name = os.path.basename(args.checkpoint_dir)
    opt.gpu_ids = [args.gpu_id] if args.gpu_id >= 0 else []
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.load_size = 256
    opt.crop_size = 256
    opt.preprocess = 'resize_and_crop'

    # Create the model
    print(f"Loading model from {args.checkpoint_dir}")
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    # Create a transform for the input images
    transform = transforms.Compose([
        transforms.Resize([opt.load_size, opt.load_size], transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Process each input image
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, f"map_{filename}")
            
            print(f"Processing {filename}...")
            input_tensor = process_image(input_path, transform)
            
            # Create input data for the model
            data = {'A': input_tensor, 'A_paths': input_path, 'B': input_tensor, 'B_paths': input_path}
            model.set_input(data)
            model.test()  # run inference
            
            # Save the output image
            fake_B = model.fake_B
            save_output_image(fake_B, output_path)

    print(f"All images processed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 