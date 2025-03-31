# Satellite Image to Map Translation with Pix2Pix

This project implements a Pix2Pix GAN model for translating satellite images to map-style images. The implementation is based on the paper [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) by Isola et al.

## Project Overview

This project demonstrates the use of Conditional Generative Adversarial Networks (cGANs) to perform image-to-image translation. Specifically, it focuses on converting satellite imagery to map representations, which can be useful for:

- Automated map generation
- Urban planning and development
- Geographic Information Systems (GIS)
- Environmental monitoring

## Features

- PyTorch implementation of Pix2Pix GAN
- Dataset preparation script for satellite-map image pairs
- Training script with validation and checkpoint saving
- Testing script to visualize results
- Support for both CPU and GPU training

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- Pillow
- matplotlib
- tqdm

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```

## GPU Acceleration

For significantly faster training, this project supports GPU acceleration via CUDA. To use a GPU:

1. Make sure you have an NVIDIA GPU
2. Install the appropriate NVIDIA drivers
3. Install a CUDA-enabled version of PyTorch:

```bash
# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

You can check if CUDA is available and working correctly:

```bash
python cuda_test.py
```

This script will provide detailed information about your PyTorch installation, CUDA availability, and GPU configuration. If CUDA is not available, it will suggest potential solutions.

## Dataset

The model is trained on the Maps dataset, which consists of pairs of satellite images and their corresponding map representations. The dataset should be structured as follows:

```
maps/
├── train/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── val/
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

Where each image contains both the satellite image (left half) and the map image (right half).

### Preparing the Dataset

Before training, you need to process the dataset into the required format:

```bash
python prepare_dataset.py --source_train "path/to/maps/train" --source_val "path/to/maps/val" --target_dir "data"
```

This will create a structured dataset in the `data` directory with separate folders for training and validation, and each containing separate subfolders for satellite and map images.

## Model Architecture

### Generator

The generator uses a U-Net architecture, which has been proven effective for image-to-image translation tasks. The U-Net consists of:

- An encoder that downsamples the input image (satellite image)
- A decoder that upsamples the encoded features
- Skip connections between corresponding layers of the encoder and decoder

### Discriminator

The discriminator follows a PatchGAN architecture, which classifies patches of the image as real or fake rather than the entire image. This encourages the generator to focus on high-frequency details.

## Training

To train the model:

```bash
python pix2pix.py
```

The script supports various options and configurations:
- Automatic CUDA detection for GPU acceleration
- Batch size and worker adjustment based on available resources
- Model checkpointing for best validation loss
- Sample image generation during training

Training progress is monitored through:
- Discriminator loss
- Generator loss (adversarial and L1)
- Validation loss
- Sample generated images

## Testing the Model

Once the model is trained, you can generate maps from satellite images:

```bash
python test_model.py --model "saved_model_10epochs.pth" --data_dir "data/val" --output_dir "results"
```

This will:
1. Load the trained generator model
2. Select random samples from the validation set
3. Generate map images from satellite inputs
4. Save side-by-side comparisons of input, generated, and target images

## Inference on Custom Images

To generate a map from a single satellite image:

```bash
python inference.py --image "path/to/satellite/image.jpg" --model "saved_model_10epochs.pth" --output "result.png"
```

This script allows you to:
- Process any satellite image (in jpg, png, or jpeg format)
- Generate a corresponding map using the trained model
- Save the original and generated images side by side

Note: For best results, use satellite images that are similar to those in the training dataset.

## Results

The model showed progressive improvement during training, with the validation loss decreasing from approximately 26.33 to 8.18 over 10 epochs. The generated maps capture the essential road structures and geographical features from the satellite images.

Sample results can be found in the `results` directory after running the test script.

## Future Improvements

- Implement data augmentation to improve model generalization
- Add support for different image resolutions
- Experiment with alternative loss functions, such as perceptual loss
- Implement a lightweight model for faster inference
- Support for transfer learning from pretrained models

## References

- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- [Pix2Pix GitHub Repository](https://github.com/phillipi/pix2pix)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

