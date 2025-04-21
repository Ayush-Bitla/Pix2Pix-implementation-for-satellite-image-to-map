# Map Generation with Pix2Pix

This README provides instructions on how to use the scripts for generating map images from satellite images using the trained Pix2Pix model.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- Pillow (PIL)
- The trained Pix2Pix model (already available in `./checkpoints/sat2map_pix2pix_50epochs/`)

## Usage

### Generate a Single Map

To generate a map from a single satellite image:

```bash
python generate_single_map.py --input_image path/to/satellite_image.jpg --output_image path/to/output_map.jpg
```

Options:
- `--input_image`: Path to the input satellite image (required)
- `--output_image`: Path to save the output map image (default: `generated_map.jpg`)
- `--checkpoint_dir`: Directory containing the trained model (default: `./checkpoints/sat2map_pix2pix_50epochs`)
- `--gpu_id`: GPU ID to use for inference, use -1 for CPU (default: -1)

### Generate Maps in Batch

To generate maps from multiple satellite images in a directory:

```bash
python generate_maps.py --input_dir path/to/satellite_images/ --output_dir path/to/output_maps/
```

Options:
- `--input_dir`: Directory containing input satellite images (required)
- `--output_dir`: Directory to save output map images (default: `./generated_maps`)
- `--checkpoint_dir`: Directory containing the trained model (default: `./checkpoints/sat2map_pix2pix_50epochs`)
- `--gpu_id`: GPU ID to use for inference, use -1 for CPU (default: -1)

## Notes

- The input satellite images should ideally be RGB images. If they're not, they will be converted to RGB.
- The scripts resize the input images to 256x256 pixels for processing.
- For best results, input images should have a similar appearance to the satellite images the model was trained on.
- The model expects satellite images (aerial photographs) and outputs map-style images.

## Example

```bash
# Generate a map from a single satellite image
python generate_single_map.py --input_image satellite_images/sample.jpg --output_image output/sample_map.jpg

# Generate maps from multiple satellite images
python generate_maps.py --input_dir satellite_images/ --output_dir output_maps/
```

## Troubleshooting

If you encounter errors:

1. Make sure the satellite images are in a standard format (JPG, PNG).
2. Check that the model checkpoint exists in the specified directory.
3. If using GPU, ensure CUDA is properly installed and the GPU is available. 