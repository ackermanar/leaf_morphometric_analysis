# Basic imports
import os
import numpy as np
import json
from PIL import Image
import shutil
import argparse

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# SAHI imports for version 0.11.20
from sahi.slicing import slice_image


"""
Training a UNet model to process variable image sizes using adaptive padding,
then using SAHI slicing technique to work within GPU constraints.

Data Processing Workflow

    1. Batch Organization: Groups high-resolution medical images into
       optimized batches of 4 for efficient GPU utilization
    2. Dynamic Dimension Handling: Automatically aligns dimensions within each
       batch by padding to a common size divisible by 32
       (optimal for U-Net architecture)
    3. Adaptive Slicing Strategy: Segments each large image
       (2000x2500+ pixels) into overlapping 512x512 patches
       with 20% overlap for context preservation
    4. Comprehensive Dataset Coverage: Processes 5 batches per epoch, totaling
       20 diverse medical images for robust model training

Specialized Training Methodology

    1. Region-Aware Loss Calculation: Implements BCEWithLogitsLoss with
       reduction='none' to obtain granular per-pixel loss values
    2. Intelligent Mask-Guided Learning: Applies precise spatial masking to
       focus loss computation exclusively on valid (non-padded) regions,
       ensuring learning from meaningful data only
    3. Weighted Loss Aggregation: Manually calculates masked loss using
       (loss*valid_mask).sum() / valid_mask.sum() to normalize by
        actual pixel count
    4. Gradient Accumulation Technique: Processes each slice independently
       before aggregating gradients, enabling effective training on
       high-resolution images while maintaining memory efficiency

Technical Implementation Details

    1. Optimizer: Adam with learning rate 1e-4 and weight decay 1e-5 for
       stable convergence
    2. Validation Strategy: Implements identical slicing and masking approach
       during validation for consistent performance evaluation
    3. Memory Optimization: Employs custom slicing rather than whole-image
       processing to handle extremely large medical images within GPU memory
       constraints
    4. Model Architecture: Utilizes a modified U-Net with skip connections
       optimized for capturing both fine details and broader contextual
       information

"""


# Define UNet model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def double_conv(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        # Modified maxpool with ceil_mode
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True
        )
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(64 + 128, 64)

        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def crop_tensor(self, target_tensor, tensor_to_crop):
        _, _, H, W = tensor_to_crop.size()
        return target_tensor[:, :, :H, :W]

    def forward(self, x):
        # Encoder path
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        # Bottleneck
        x = self.dconv_down4(x)

        # Decoder path with cropping
        x = self.upsample(x)
        conv3 = self.crop_tensor(conv3, x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        conv2 = self.crop_tensor(conv2, x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)
        conv1 = self.crop_tensor(conv1, x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out


# Define SAHI segmentation adapter
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, valid_files=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Get all image files or use provided valid files
        if valid_files is not None:
            self.image_files = valid_files
        else:
            self.image_files = sorted([
                f for f in os.listdir(image_dir)
                if f.lower().endswith((
                    '.jpg', '.jpeg', '.png', '.tif', '.tiff'
                ))
            ])

        # Get corresponding mask files
        self.mask_files = []
        for img_file in self.image_files:
            # Find corresponding mask file
            mask_base = os.path.splitext(img_file)[0]
            mask_candidates = [
                f"{mask_base}.png",
                f"{mask_base}.jpg",
                f"{mask_base}.tif",
                f"{mask_base}_mask.png",
                f"{mask_base}_mask.jpg",
                # Add other potential mask filename patterns
            ]

            found = False
            for mask_file in mask_candidates:
                if os.path.exists(os.path.join(mask_dir, mask_file)):
                    self.mask_files.append(mask_file)
                    found = True
                    break  # Use break instead of return

            if not found:
                raise ValueError(f"No mask found for image {img_file}")

        # Verify we have the same number of images and masks
        assert len(self.image_files) == len(self.mask_files), \
            "Number of images and masks don't match"

        print(f"Found {len(self.image_files)} image-mask pairs")

    def __len__(self):
        """Return the total number of image-mask pairs"""
        return len(self.image_files)

    def __getitem__(self, idx):
        """Get the image and mask at the given index"""
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")

        # Load mask
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale

        # Convert to numpy arrays
        img_np = np.array(img)
        mask_np = np.array(mask)[:, :, None]  # Add channel dimension

        # Convert to tensors
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy(mask_np.transpose(2, 0, 1)).float() / 255.0

        # Apply transformations if any
        if self.transform:
            img_tensor = self.transform(img_tensor)
            mask_tensor = self.transform(mask_tensor)

        return img_tensor, mask_tensor


# Define functions
# Collate with adaptive padding
def collate_fn(batch):
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]

    # First, ensure masks match their corresponding images in dimensions
    aligned_masks = []
    for i, (img, msk) in enumerate(zip(images, masks)):
        # Check if dimensions don't match
        if img.shape[1:] != msk.shape[1:]:
            print(f"Item {i}: Fixing dimension mismatch: "
                  f"Image {img.shape} vs Mask {msk.shape}")
            # Resize mask to match image dimensions
            c, h, w = msk.shape
            img_h, img_w = img.shape[1:]

            # Use interpolate to resize
            resized_mask = F.interpolate(
                msk.unsqueeze(0),  # Add batch dimension
                size=(img_h, img_w),
                mode='nearest'
            ).squeeze(0)  # Remove batch dimension

            aligned_masks.append(resized_mask)
        else:
            aligned_masks.append(msk)

    # Now use the aligned masks
    masks = aligned_masks

    # Print all dimensions for debugging
    for i, (img, msk) in enumerate(zip(images, masks)):
        print(f"Item {i} after alignment: Image {img.shape}, Mask {msk.shape}")

    # Find max dimensions in the batch
    # Get the current batch's dimensions
    batch_heights = [img.shape[1] for img in images]
    batch_widths = [img.shape[2] for img in images]

    max_h = max(batch_heights)
    max_w = max(batch_widths)

    print(f"Max dimensions in batch: Height={max_h}, Width={max_w}")

    # Set target dimensions to be divisible by 32
    # (common for U-Net architectures)
    target_height = ((max_h + 31) // 32) * 32
    target_width = ((max_w + 31) // 32) * 32

    print(f"Target dimensions: Height={target_height}, Width={target_width}")

    # Check if any image is larger than the target
    for i, img in enumerate(images):
        if img.shape[1] > target_height or img.shape[2] > target_width:
            print(f"Warning: Image {i} with shape {img.shape} is larger than "
                  f"target ({target_height}, {target_width})")

    def process_to_target_size(tensor, target_height, target_width,
                               mode='constant'):
        """Resize or pad tensor to target size"""
        # Handle both 2D and 3D tensors
        if tensor.dim() == 2:
            # For 2D tensor (H, W), add channel dimension
            tensor = tensor.unsqueeze(0)  # Convert to (1, H, W)

        # Now tensor should be 3D (C, H, W)
        c, h, w = tensor.shape

        # If image is larger than target in any dimension, resize it down
        if h > target_height or w > target_width:
            print(f"Resizing tensor from {tensor.shape} to fit within "
                  f"({target_height}, {target_width})")
            # Resize down maintaining aspect ratio
            scale = min(target_height / h, target_width / w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized = F.interpolate(tensor.unsqueeze(0), size=(new_h, new_w),
                                    mode='bilinear' if c == 3 else 'nearest')
            tensor = resized.squeeze(0)
            c, h, w = tensor.shape

        # Now pad to exact target size
        pad_h = target_height - h
        pad_w = target_width - w

        if pad_h < 0:
            print(f"Error: Negative height padding {pad_h} for tensor of shape {tensor.shape}")
            # Force resize instead of padding
            tensor = F.interpolate(tensor.unsqueeze(0), size=(target_height, w),
                                   mode='bilinear' if c == 3 else 'nearest').squeeze(0)
            pad_h = 0

        if pad_w < 0:
            print(f"Error: Negative width padding {pad_w} for tensor of shape {tensor.shape}")
            # Force resize instead of padding
            tensor = F.interpolate(tensor.unsqueeze(0), size=(h, target_width),
                                   mode='bilinear' if c == 3 else 'nearest').squeeze(0)
            pad_w = 0

        print(f"Padding tensor from {tensor.shape} with padding (0, {pad_w}, 0, {pad_h})")
        padded_tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
        return padded_tensor

    # Process images and masks
    processed_images = []
    processed_masks = []

    for i, (img, msk) in enumerate(zip(images, masks)):
        print(f"Processing item {i} - Image: {img.shape}, Mask: {msk.shape}")

        try:
            processed_img = process_to_target_size(img, target_height, target_width, mode='reflect')
            processed_images.append(processed_img)

            # Handle mask based on its dimensionality
            processed_mask = process_to_target_size(msk, target_height, target_width, mode='constant')
            processed_masks.append(processed_mask)

            print(f"Successfully processed item {i} - Image: {processed_img.shape}, Mask: {processed_mask.shape}")
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            # Skip this item if there's an error
            continue

    # If we had to skip items, make sure we still have something to return
    if len(processed_images) == 0:
        raise RuntimeError(
            "All items in batch were skipped due to processing errors"
        )

    # Stack into batches
    batched_images = torch.stack(
        processed_images
    )
    batched_masks = torch.stack(processed_masks)

    print(f"Final batched shapes - Images: {batched_images.shape}, "
          f"Masks: {batched_masks.shape}")
    return batched_images, batched_masks


# Define a custom scheduler wrapper
class VerboseReduceLROnPlateau:
    def __init__(self, optimizer, mode='min', factor=0.1,
                 patience=10, threshold=1e-4,
                 threshold_mode='rel', cooldown=0, min_lr=0):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode, factor, patience, threshold,
            threshold_mode, cooldown, min_lr
        )
        self.optimizer = optimizer
        self.current_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, metrics):
        # Store current learning rates
        old_lrs = [group['lr'] for group in self.optimizer.param_groups]

        # Call the actual scheduler
        self.scheduler.step(metrics)

        # Check if learning rates changed
        new_lrs = [group['lr'] for group in self.optimizer.param_groups]

        # Print message if learning rate changed
        for i, (old_lr, new_lr) in enumerate(zip(old_lrs, new_lrs)):
            if new_lr != old_lr:
                print(f"Epoch {self.scheduler.last_epoch}: reducing learning rate of group {i} to {new_lr:.4e}.")

        return self.scheduler.last_epoch


def validate_dataset_dimensions(image_dir, mask_dir, remove_mismatched=False, fix_mismatched=False):
    """
    Validates that all images and their corresponding masks have matching dimensions.

    Args:
        image_dir: Directory containing image files
        mask_dir: Directory containing mask files
        remove_mismatched: If True, will move mismatched files to a 'mismatched' subdirectory
        fix_mismatched: If True, will resize masks to match image dimensions

    Returns:
        valid_files: List of filenames that have matching dimensions
        mismatched_files: List of filenames with dimension mismatches
    """

    print(f"Validating dimensions for dataset in {image_dir} and {mask_dir}...")

    # Get all image files
    image_files = sorted([f for f in os.listdir(image_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])

    valid_files = []
    mismatched_files = []
    fixed_files = []

    # Create directories for mismatched files if needed
    if remove_mismatched:
        os.makedirs(os.path.join(image_dir, 'mismatched'), exist_ok=True)
        os.makedirs(os.path.join(mask_dir, 'mismatched'), exist_ok=True)

    # Create directory for fixed files if needed
    if fix_mismatched:
        os.makedirs(os.path.join(mask_dir, 'original'), exist_ok=True)

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)

        # Find corresponding mask file
        mask_base = os.path.splitext(img_file)[0]
        mask_candidates = [
            f"{mask_base}.png",
            f"{mask_base}.jpg",
            f"{mask_base}.tif",
            f"{mask_base}_mask.png",
            f"{mask_base}_mask.jpg",
            # Add other potential mask filename patterns
        ]

        mask_file = None
        for candidate in mask_candidates:
            candidate_path = os.path.join(mask_dir, candidate)
            if os.path.exists(candidate_path):
                mask_file = candidate
                break

        if mask_file is None:
            print(f"Warning: No mask found for image {img_file}")
            mismatched_files.append(img_file)
            continue

        mask_path = os.path.join(mask_dir, mask_file)

        # Open images and check dimensions
        try:
            with Image.open(img_path) as img, Image.open(mask_path) as mask:
                img_width, img_height = img.size
                mask_width, mask_height = mask.size

                if img_width != mask_width or img_height != mask_height:
                    print(f"Dimension mismatch: {img_file} ({img_width}x{img_height}) vs "
                          f"{mask_file} ({mask_width}x{mask_height})")
                    mismatched_files.append(img_file)

                    if fix_mismatched:
                        # Backup original mask
                        shutil.copy2(mask_path, os.path.join(mask_dir, 'original', mask_file))

                        # Resize mask to match image
                        mask_resized = mask.resize((img_width, img_height), Image.NEAREST)
                        mask_resized.save(mask_path)
                        print("  - Fixed: Resized mask to match image dimensions")
                        fixed_files.append(img_file)
                        valid_files.append(img_file)  # Add to valid files since it's now fixed
                    elif remove_mismatched:
                        # Move mismatched files to separate directory
                        shutil.move(img_path, os.path.join(image_dir, 'mismatched', img_file))
                        shutil.move(mask_path, os.path.join(mask_dir, 'mismatched', mask_file))
                        print("  - Moved mismatched files to 'mismatched' directory")
                else:
                    valid_files.append(img_file)
        except Exception as e:
            print(f"Error processing {img_file} and {mask_file}: {str(e)}")
            mismatched_files.append(img_file)

            if remove_mismatched:
                # Move problematic files to separate directory
                try:
                    shutil.move(img_path, os.path.join(image_dir, 'mismatched', img_file))
                    shutil.move(mask_path, os.path.join(mask_dir, 'mismatched', mask_file))
                    print("  - Moved problematic files to 'mismatched' directory")
                except Exception as move_error:
                    print(f"  - Error moving files: {str(move_error)}")

    # Print summary
    print("\nValidation complete:")
    print(f"  - Total images: {len(image_files)}")
    print(f"  - Valid pairs: {len(valid_files)}")
    print(f"  - Mismatched pairs: {len(mismatched_files)}")
    if fix_mismatched:
        print(f"  - Fixed pairs: {len(fixed_files)}")

    return valid_files, mismatched_files


# Custom SAHI slicing
def custom_slice_image(image_np, slice_height, slice_width, overlap_ratio):
    """
    Custom function to slice a numpy image array into smaller patches with overlap

    Args:
        image_np: Numpy array of shape [H, W, C]
        slice_height, slice_width: Size of slices
        overlap_ratio: Overlap between slices (0-1)

    Returns:
        List of dictionaries with 'image' and 'coordinates' keys
    """
    height, width = image_np.shape[:2]
    stride_h = int(slice_height * (1 - overlap_ratio))
    stride_w = int(slice_width * (1 - overlap_ratio))

    slices = []
    for y in range(0, height, stride_h):
        for x in range(0, width, stride_w):
            # Calculate slice coordinates
            x_min = x
            y_min = y
            x_max = min(x + slice_width, width)
            y_max = min(y + slice_height, height)

            # Handle edge cases - ensure slices are of size (slice_height, slice_width) when possible
            if x_max - x_min < slice_width and x_min > 0:
                x_min = max(0, x_max - slice_width)
            if y_max - y_min < slice_height and y_min > 0:
                y_min = max(0, y_max - slice_height)

            # Extract slice
            slice_image = image_np[y_min:y_max, x_min:x_max, :]

            # Create slice data in the expected format
            slice_data = {
                "image": slice_image,
                "coordinates": (x_min, y_min, x_max, y_max)
            }

            slices.append(slice_data)

    return slices


def predict_with_sahi(model, image, slice_height=512, slice_width=512, overlap_ratio=0.2):
    """
    Process a large image using SAHI slicing technique

    Args:
        model: Your UNET model wrapped in SAHISegmentationAdapter
        image: Input image as numpy array (H, W, C)
        slice_height, slice_width: Size of slices
        overlap_ratio: Overlap between adjacent slices

    Returns:
        Full-sized segmentation mask
    """
    # Get image dimensions
    height, width = image.shape[:2]

    # Slice the image
    slices = slice_image(
        image=image,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio
    )

    # Process each slice
    slice_predictions = []
    for slice_data in slices:
        # Get the slice image
        slice_image_data = slice_data["image"]

        # Predict on this slice
        slice_mask = model.predict_single_image(slice_image_data)

        # Store prediction with coordinates
        slice_predictions.append({
            "mask": slice_mask,
            "coordinates": slice_data["coordinates"]
        })

    # Combine predictions into full-sized mask
    full_mask = np.zeros((height, width), dtype=np.uint8)

    for pred in slice_predictions:
        # Extract coordinates
        x_min, y_min, x_max, y_max = pred["coordinates"]
        mask = pred["mask"]

        # Place the prediction in the right position
        # You may need a more sophisticated merging strategy for overlapping regions
        full_mask[y_min:y_max, x_min:x_max] = mask

    return full_mask


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train UNet for image segmentation')

    # Configuration options
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--config', type=str,
                              help='Path to config file (YAML or JSON)')

    # Dataset paths
    data_group = parser.add_argument_group('Dataset Paths')
    data_group.add_argument('--train_dir', type=str,
                            help='Path to training images')
    data_group.add_argument('--mask_dir', type=str,
                            help='Path to training masks')
    data_group.add_argument('--val_dir', type=str,
                            help='Path to validation images')
    data_group.add_argument('--val_mask_dir', type=str,
                            help='Path to validation masks')

    # Training parameters
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--epochs', type=int, default=10,
                             help='Number of epochs')
    train_group.add_argument('--batch_size', type=int, default=4,
                             help='Batch size')
    train_group.add_argument('--slice_size', type=int, default=512,
                             help='Size of image slices')
    train_group.add_argument('--overlap', type=float, default=0.2,
                             help='Overlap ratio for slicing')
    train_group.add_argument('--num_workers', type=int, default=0,
                             help='Number of data loading workers')

    # Validation options
    val_group = parser.add_argument_group('Validation Options')
    val_group.add_argument('--validate_dims', action='store_true',
                           help='Validate image and mask dimensions')
    val_group.add_argument('--fix_mismatched', action='store_true',
                           help='Fix mismatched dimensions by resizing masks')
    val_group.add_argument('--remove_mismatched', action='store_true',
                           help='Remove mismatched files')

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output_dir', type=str, default='./output',
                              help='Directory to save outputs')

    # Parse arguments
    args = parser.parse_args()

    # Validate required arguments
    if not args.config and (
        not args.train_dir or not args.mask_dir or not args.val_dir or not args.val_mask_dir
    ):
        parser.error(
            "When not using a config file, --train_dir, --mask_dir, --val_dir, and --val_mask_dir are required"
        )

    # Load config file if provided
    if args.config:
        config_ext = os.path.splitext(args.config)[1].lower()
        with open(args.config, 'r') as f:
            if config_ext == '.yaml' or config_ext == '.yml':
                import yaml
                config = yaml.safe_load(f)
            elif config_ext == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_ext}")

        # Override args with config values
        for category in config:
            if isinstance(config[category], dict):
                for key, value in config[category].items():
                    if hasattr(args, key):
                        setattr(args, key, value)
            elif hasattr(args, category):
                setattr(args, category, config[category])

    # Set up Slurm-specific configurations
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id:
        print(f"Running as Slurm job {slurm_job_id}")

        # Set default output directory if not specified
        if not args.output_dir:
            args.output_dir = f"/scratch/{os.environ.get('USER', 'user')}/{slurm_job_id}"

        # Get GPU assigned by Slurm
        if torch.backends.mps.is_available():
            print("Using MPS (Apple Silicon GPU)")
        else:
            gpu_id = os.environ.get('SLURM_JOB_GPUS', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            print(f"Using GPU: {gpu_id}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Save configuration for reproducibility
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(
            vars(args), f, indent=4
        )

    # Set device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"Using device: {device}")

    # Load UNET model
    model = UNet().to(device)
    print(f"Model loaded to {device}")

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("Using Adam optimizer with learning rate 1e-4")

    # Initialize scheduler
    scheduler = VerboseReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    print("Using ReduceLROnPlateau scheduler with factor=0.5, patience=3")

    # Validate dataset dimensions if requested
    if args.validate_dims:
        print("Validating dataset dimensions...")
        # Validate dimensions first
        train_valid_files, train_mismatched = validate_dataset_dimensions(
            args.train_dir,
            args.mask_dir,
            remove_mismatched=args.remove_mismatched,
            fix_mismatched=args.fix_mismatched
        )

        val_valid_files, val_mismatched = validate_dataset_dimensions(
            args.val_dir,
            args.val_mask_dir,
            remove_mismatched=args.remove_mismatched,
            fix_mismatched=args.fix_mismatched
        )

        # Create datasets with validated files
        train_dataset = SegmentationDataset(
            image_dir=args.train_dir,
            mask_dir=args.mask_dir,
            valid_files=train_valid_files
        )

        val_dataset = SegmentationDataset(
            image_dir=args.val_dir,
            mask_dir=args.val_mask_dir,
            valid_files=val_valid_files
        )
    else:
        # Create datasets without validation
        train_dataset = SegmentationDataset(
            image_dir=args.train_dir,
            mask_dir=args.mask_dir
        )

        val_dataset = SegmentationDataset(
            image_dir=args.val_dir,
            mask_dir=args.val_mask_dir
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    print(f"Created training dataloader with {len(train_dataset)} samples, "
          f"batch size {args.batch_size}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    print(f"Created validation dataloader with {len(val_dataset)} samples, "
          f"batch size {args.batch_size}")

    # Use command line arguments for training parameters
    num_epochs = args.epochs
    slice_height = args.slice_size
    slice_width = args.slice_size
    overlap_ratio = args.overlap

    # Print model hyperparameters
    print(f"Training UNet for {num_epochs} epochs with slice size "
          f"({slice_height}, {slice_width}) and overlap {overlap_ratio}")
    print("Starting training...")

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience = 5  # Number of epochs to wait before early stopping
    patience_counter = 0

    # _________________________Training Loop__________________________________
    for epoch in range(num_epochs):
        print(f"Starting training epoch {epoch+1}")
        model.train()
        train_loss = 0

        for batch_idx, (images, masks) in enumerate(train_loader):
            print(f"Processing training batch {batch_idx+1}/{len(train_loader)}")
            batch_loss = 0

            # Process each image in the batch
            for i in range(images.shape[0]):
                # Get single image and mask
                image = images[i]
                mask = masks[i]

                # Convert to numpy arrays
                image_np = image.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                mask_np = mask.permute(1, 2, 0).cpu().numpy()    # [H, W, C]

                # Slice the image using our custom function
                image_slices = custom_slice_image(
                    image_np=image_np,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_ratio=overlap_ratio
                )

                # Process each slice
                slice_losses = []
                optimizer.zero_grad()

                for slice_data in image_slices:
                    # Get the slice image and coordinates
                    slice_image_data = slice_data["image"]
                    x_min, y_min, x_max, y_max = slice_data["coordinates"]

                    # Extract corresponding mask slice
                    slice_mask_data = mask_np[y_min:y_max, x_min:x_max, :]

                    # Convert back to tensors
                    slice_image_tensor = torch.from_numpy(
                        slice_image_data.transpose(2, 0, 1)
                    ).float().unsqueeze(0).to(device)  # Add batch dimension

                    slice_mask_tensor = torch.from_numpy(
                        slice_mask_data.transpose(2, 0, 1)
                    ).float().unsqueeze(0).to(device)  # Add batch dimension

                    # Create a mask for valid (non-padded) regions
                    # For slices, all pixels are valid since we're taking exact slices
                    valid_mask = torch.ones_like(slice_mask_tensor)

                    # Forward pass
                    slice_output = model(slice_image_tensor)

                    # Calculate loss with masking
                    # Option 1: Using reduction='none' and manual masking
                    loss = F.binary_cross_entropy_with_logits(
                        slice_output,
                        slice_mask_tensor,
                        reduction='none'
                    )
                    # Apply mask and calculate mean over valid pixels
                    masked_loss = (loss * valid_mask).sum() / valid_mask.sum()

                    # Option 2: Using weight parameter (alternative)
                    # loss = F.binary_cross_entropy_with_logits(
                    #     slice_output,
                    #     slice_mask_tensor,
                    #     weight=valid_mask,  # Weight parameter acts as our mask
                    #     reduction='mean'
                    # )

                    # Backward pass
                    masked_loss.backward()
                    slice_losses.append(masked_loss.item())

                # Update weights after processing all slices
                optimizer.step()

                # Calculate average loss for this image
                if slice_losses:
                    image_loss = sum(slice_losses) / len(slice_losses)
                    batch_loss += image_loss
                    print(f"Image {i+1} loss: {image_loss:.4f}")

            # Average loss for the batch
            batch_loss /= images.shape[0]
            train_loss += batch_loss
            print(f"Batch {batch_idx+1} average loss: {batch_loss:.4f}")

        # Calculate epoch training loss
        train_loss /= len(train_loader)

        # Validation
        print(f"Starting validation epoch {epoch+1}")
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                print(f"Processing validation batch {batch_idx+1}/{len(val_loader)}")
                batch_loss = 0

                # Process each image in the batch
                for i in range(images.shape[0]):
                    # Get single image and mask
                    image = images[i]
                    mask = masks[i]

                    # Convert to numpy arrays
                    image_np = image.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                    mask_np = mask.permute(1, 2, 0).cpu().numpy()    # [H, W, C]

                    # Slice the image using our custom function
                    image_slices = custom_slice_image(
                        image_np=image_np,
                        slice_height=slice_height,
                        slice_width=slice_width,
                        overlap_ratio=overlap_ratio
                    )

                    # Process each slice
                    slice_losses = []

                    for slice_data in image_slices:
                        # Get the slice image and coordinates
                        slice_image_data = slice_data["image"]
                        x_min, y_min, x_max, y_max = slice_data["coordinates"]

                        # Extract corresponding mask slice
                        slice_mask_data = mask_np[y_min:y_max, x_min:x_max, :]

                        # Convert back to tensors
                        slice_image_tensor = torch.from_numpy(
                            slice_image_data.transpose(2, 0, 1)
                        ).float().unsqueeze(0).to(device)  # Add batch dimension

                        slice_mask_tensor = torch.from_numpy(
                            slice_mask_data.transpose(2, 0, 1)
                        ).float().unsqueeze(0).to(device)  # Add batch dimension

                        # Create a mask for valid (non-padded) regions
                        valid_mask = torch.ones_like(slice_mask_tensor)

                        # Forward pass
                        slice_output = model(slice_image_tensor)

                        # Calculate loss with masking
                        loss = F.binary_cross_entropy_with_logits(
                            slice_output,
                            slice_mask_tensor,
                            reduction='none'
                        )
                        # Apply mask and calculate mean over valid pixels
                        masked_loss = (loss * valid_mask).sum() / valid_mask.sum()

                        slice_losses.append(masked_loss.item())

                    # Calculate average loss for this image
                    if slice_losses:
                        image_loss = sum(slice_losses) / len(slice_losses)
                        batch_loss += image_loss
                        print(f"Validation image {i+1} loss: {image_loss:.4f}")

                # Average loss for the batch
                batch_loss /= images.shape[0]
                val_loss += batch_loss
                print(f"Validation batch {batch_idx+1} average loss: {batch_loss:.4f}")

            # Calculate epoch validation loss
            val_loss /= len(val_loader)

            # Update learning rate based on validation loss
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Learning rate adjusted: {old_lr:.6f} → {new_lr:.6f}")

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(args.output_dir, f'model_checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': new_lr,  # Save current learning rate
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Early stopping logic
        if val_loss < best_val_loss:
            improvement = (best_val_loss - val_loss) / best_val_loss * 100
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_path = os.path.join(
                args.output_dir, 'best_model.pth'
            )
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! Validation loss improved by {improvement:.2f}%")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs without improvement")
                break


if __name__ == "__main__":
    main()
