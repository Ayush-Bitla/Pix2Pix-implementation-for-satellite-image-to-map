#!/usr/bin/env python
# generate_single_map.py
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
    parser = argparse.ArgumentParser(description='Generate a map from a satellite image using trained pix2pix model')
    parser.add_argument('--input_image', type=str, required=True, help='Path to input satellite image')
    parser.add_argument('--output_image', type=str, default='generated_map.jpg', help='Path to save output map image')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/sat2map_pix2pix_50epochs', 
                        help='Directory with trained model checkpoint')
    parser.add_argument('--gpu_id', type=int, default=-1, help='GPU ID (-1 for CPU)')
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_image):
        print(f"Error: Input file {args.input_image} does not exist.")
        return

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_image)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

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

    # Create a transform for the input image
    transform = transforms.Compose([
        transforms.Resize([opt.load_size, opt.load_size], transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Process the input image
    print(f"Processing image: {args.input_image}")
    input_tensor = process_image(args.input_image, transform)
    
    # Create input data for the model
    data = {'A': input_tensor, 'A_paths': args.input_image, 'B': input_tensor, 'B_paths': args.input_image}
    model.set_input(data)
    model.test()  # run inference
    
    # Save the output image
    fake_B = model.fake_B
    save_output_image(fake_B, args.output_image)
    print(f"Processing complete. Output image saved to {args.output_image}")


if __name__ == "__main__":
    main() 