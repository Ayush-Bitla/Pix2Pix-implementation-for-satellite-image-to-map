# ml_project.py
import os
import tarfile
import argparse
import random
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch


def extract_dataset(tar_path, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=extract_path)
        tqdm.write(f"✅ Dataset extracted to: {extract_path}")
    else:
        tqdm.write("ℹ️ Dataset already extracted.")


def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((286 * 2, 286))  # 572 x 286
    max_crop_x = 572 - 512
    max_crop_y = 286 - 256
    x = random.randint(0, max_crop_x // 2)
    y = random.randint(0, max_crop_y)
    img = img.crop((x * 2, y, x * 2 + 512, y + 256))
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def preprocess_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tqdm.write(f"📦 Preprocessing training images from: {input_dir}")
    for fname in tqdm(os.listdir(input_dir)):
        if fname.endswith(('.jpg', '.png')):
            img_path = os.path.join(input_dir, fname)
            img = preprocess_image(img_path)
            img.save(os.path.join(output_dir, fname))
    tqdm.write(f"✅ Preprocessing complete. Saved to: {output_dir}")


def preprocess_val_image(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((286 * 2, 286))
        x = (286 - 256) // 2
        y = (286 - 256) // 2
        img = img.crop((x * 2, y, x * 2 + 512, y + 256))
        return img
    except OSError as e:
        tqdm.write(f"❌ Error processing {img_path}: {e}")
        return None


def preprocess_val_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tqdm.write(f"📦 Preprocessing validation images from: {input_dir}")
    for fname in tqdm(os.listdir(input_dir)):
        if fname.endswith(('.jpg', '.png')):
            img_path = os.path.join(input_dir, fname)
            img = preprocess_val_image(img_path)
            if img is not None:
                img.save(os.path.join(output_dir, fname))
    tqdm.write(f"✅ Validation preprocessing done: {output_dir}")


def run_pix2pix_training(dataroot, name="sat2map_pix2pix_50epochs"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tqdm.write(f"🚀 Training on device: {device}")

    if device == "cuda":
       os.system(f"python pytorch-CycleGAN-and-pix2pix/train.py \
            --dataroot {dataroot} \
            --name {name} \
            --model pix2pix \
            --direction AtoB \
            --batch_size 1 \
            --n_epochs 100 \
            --n_epochs_decay 0 \
            --display_freq 100 \
            --print_freq 100 \
            --gpu_ids 0")
    else:
        tqdm.write("⚠️ CUDA not available. Training skipped.")


def visualize_sample(image_path, title):
    img = Image.open(image_path)
    plt.figure(figsize=(8, 4))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Pix2Pix Dataset Preprocessing and Training")
    parser.add_argument('--tar_path', type=str, default='maps.tar.gz', help='Path to the .tar.gz dataset')
    parser.add_argument('--dataset_path', type=str, default='datasets/maps', help='Directory to extract and process dataset')
    parser.add_argument('--train', action='store_true', help='Run Pix2Pix training after preprocessing')
    parser.add_argument('--visualize', action='store_true', help='Show a sample preprocessed training image')
    args = parser.parse_args()

    extract_dataset(args.tar_path, args.dataset_path)

    preprocess_folder(os.path.join(args.dataset_path, 'train'), os.path.join(args.dataset_path, 'train_processed'))
    preprocess_val_folder(os.path.join(args.dataset_path, 'val'), os.path.join(args.dataset_path, 'val_processed'))

    if args.train:
        run_pix2pix_training(args.dataset_path)

    if args.visualize:
        train_proc_path = os.path.join(args.dataset_path, 'train_processed')
        if os.listdir(train_proc_path):
            sample_path = os.path.join(train_proc_path, random.choice(os.listdir(train_proc_path)))
            visualize_sample(sample_path, "Sample Preprocessed Training Image")
        else:
            tqdm.write("❌ No images found in train_processed for visualization.")


if __name__ == "__main__":
    main()
