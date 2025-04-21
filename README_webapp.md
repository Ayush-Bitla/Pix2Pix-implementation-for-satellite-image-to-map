# Satellite to Map Web Application

This web application provides a user-friendly interface for generating map images from satellite/aerial photographs using a Pix2Pix GAN model.

## Features

- Simple, intuitive web interface
- Upload satellite images in JPG, JPEG, or PNG format
- Preview uploaded images before processing
- View side-by-side comparison of input and output images
- Download generated map images

## Requirements

- Python 3.6+
- Flask
- PyTorch
- torchvision
- Pillow (PIL)
- NumPy 1.26.4 (important for compatibility)
- The trained Pix2Pix model (should be in `./checkpoints/sat2map_pix2pix_50epochs/`)

## Installation

### Option 1: Automated Setup (Recommended)

Use the setup script to create a virtual environment with compatible dependencies:

```bash
python setup.py
```

This will:
1. Create a virtual environment (.venv)
2. Install NumPy 1.26.4 for compatibility with PyTorch
3. Install all other dependencies
4. Create necessary directories
5. Provide instructions to run the application

### Option 2: Manual Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Unix/Linux/Mac: `source .venv/bin/activate`

3. Install compatible NumPy first:
   ```bash
   pip install numpy==1.26.4
   ```

4. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Make sure you have the trained model in place:
   - The model should be in `./checkpoints/sat2map_pix2pix_50epochs/latest_net_G.pth`

## Usage

1. Activate the virtual environment (if not already activated):
   - Windows: `.venv\Scripts\activate`
   - Unix/Linux/Mac: `source .venv/bin/activate`

2. Start the Flask server:
   ```bash
   python app.py
   ```

3. Open a web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

4. Upload a satellite/aerial image using the web interface

5. The application will process the image and display the generated map

6. You can download the generated map or generate more maps with different images

## Folder Structure

- `app.py` - Flask web application
- `setup.py` - Setup script for environment configuration
- `templates/` - HTML templates
  - `index.html` - Home page with upload form
  - `result.html` - Results page showing original and generated images
- `static/` - Static files
  - `uploads/` - Stores uploaded satellite images
  - `generated/` - Stores generated map images
- `checkpoints/` - Contains the trained Pix2Pix model

## Notes

- For best results, use satellite images similar to Google Maps aerial view
- The model works best on urban areas, roads, and clear land features
- Processing time may vary depending on your computer's specifications
- Images are resized to 256x256 pixels during processing

## Troubleshooting

### NumPy Compatibility Issues

If you see errors about NumPy compatibility:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x...
```

This is because PyTorch 2.2.0 requires NumPy 1.x, but you have NumPy 2.x installed. To fix this:

1. Uninstall NumPy and reinstall the compatible version:
   ```bash
   pip uninstall -y numpy
   pip install numpy==1.26.4
   ```

2. Alternatively, run the `setup.py` script which will set up a clean environment with compatible versions.

### Other Issues

- Make sure the model file exists in the correct location
- Check that all dependencies are installed correctly
- Ensure your image is in a supported format (JPG, JPEG, or PNG)
- Try using different satellite images if you're getting poor results 