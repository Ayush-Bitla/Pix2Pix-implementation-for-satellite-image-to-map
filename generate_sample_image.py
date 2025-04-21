import numpy as np
from PIL import Image, ImageDraw
import os

# Ensure directory exists
os.makedirs('static', exist_ok=True)

# Create a blank image (512x512)
width, height = 512, 512
image = np.zeros((height, width, 3), dtype=np.uint8)

# Set background color (dark green for grass/trees)
image[:, :] = (30, 100, 30)

# Add some features that look like a satellite view
# Main road (gray)
image[200:312, 50:462] = (120, 120, 120)

# Buildings (various grays and whites)
# Downtown area
for i in range(10):
    for j in range(10):
        if np.random.random() > 0.3:  # Some empty lots
            bx, by = 50 + i*30, 50 + j*15
            bw, bh = np.random.randint(15, 28), np.random.randint(8, 14)
            color = np.random.randint(150, 250, size=3)
            image[by:by+bh, bx:bx+bw] = color

# Residential area
for i in range(15):
    for j in range(10):
        if np.random.random() > 0.4:  # Some empty lots
            bx, by = 350 + i*10, 50 + j*20
            bw, bh = np.random.randint(5, 9), np.random.randint(10, 18)
            color = (200, 200, 200)
            image[by:by+bh, bx:bx+bw] = color

# Parks and fields (bright green)
image[350:450, 100:250] = (60, 180, 60)
image[100:150, 350:450] = (60, 180, 60)

# Water (blue)
image[350:500, 350:500] = (30, 100, 170)

# Convert to PIL image
img = Image.fromarray(image)

# Add some noise to make it look more realistic
draw = ImageDraw.Draw(img)
for _ in range(5000):
    x = np.random.randint(0, width)
    y = np.random.randint(0, height)
    r, g, b = image[y, x]
    
    # Add slight variation to pixel color
    variation = np.random.randint(-20, 20, size=3)
    r = max(0, min(255, r + variation[0]))
    g = max(0, min(255, g + variation[1]))
    b = max(0, min(255, b + variation[2]))
    
    draw.point((x, y), fill=(r, g, b))

# Save the image
img.save('static/sample_satellite.jpg')
print("Sample satellite image created at: static/sample_satellite.jpg")

# Create a version with proper dimensions for the model (512x256)
img_resized = img.resize((512, 256), Image.LANCZOS)
img_resized.save('static/sample_satellite_model_size.jpg')
print("Model-ready sample created at: static/sample_satellite_model_size.jpg") 