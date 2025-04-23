# crop script
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from PIL import Image, ImageOps
import sys
import json
import cv2
import argparse

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def create_transform_matrix(translation, rotation):
    """Create a 4x4 transform matrix from translation and rotation."""
    transform = np.eye(4)
    transform[:3, 3] = translation
    transform[:3, :3] = rotation
    return transform.tolist()

def apply_mask(img, mask):
    # binary mask (make sure it's reshaped to match)
    mask = mask / 255
    masked_img = img * mask.reshape(*mask.shape, 1)
    return masked_img.astype(np.uint8)
    
def get_multiview_sample(image_path, mask_path, timestep: int, from_cameras: None):
    # for some timestep, retrieve images from each camera (or from from_cameras)
    # at a specific timestep
    # Get list of camera directories from image path
    camera_dirs = [d for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))]
    
    # If specific cameras requested, filter to only those
    # Otherwise use all camera directories found
    if from_cameras is not None:
        camera_dirs = [d for d in camera_dirs if d in from_cameras]
        
    images = {}
    masks = {}
    
    for camera in camera_dirs:
        # Construct paths for this timestep
        img_file = f"{str(timestep).zfill(4)}_img.jpg"
        mask_file = f"{str(timestep).zfill(4)}_img_fmask.png"
        
        img_path = os.path.join(image_path, camera, img_file)
        mask_path_full = os.path.join(mask_path, camera, mask_file)
        
        # Load if files exist
        if os.path.exists(img_path) and os.path.exists(mask_path_full):
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path_full))
            
            # Store in dictionaries
            images[camera] = img
            masks[camera] = mask
            
    return images, masks

# intrinsic update functions + cropping
def get_bbox_from_annot(annot):
    """Extract bbox from annotation and convert to [x1, y1, x2, y2] format."""
    bbox = annot['annots'][0]['bbox']  # Assuming single person
    return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]

def get_bbox_center_and_size(bbox):
    """Get center point and size of bbox."""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return (center_x, center_y), (width, height)

def update_intrinsics(K, crop_x, crop_y, original_width, original_height):
    """Update intrinsic matrix for the crop."""
    # Create new intrinsic matrix
    K_new = K.copy()
    
    if crop_x > 0:
        K_new[0, 2] = K[0, 2] - crop_x
    if crop_y > 0:
        K_new[1, 2] = K[1, 2] - crop_y
    
    return K_new

def process_images_and_intrinsics(base_dir, image_id="0005", scale=0.5):
    """Process all images and update intrinsics based on bounding boxes."""
    # image_id: number(XXXX)
    # Load camera intrinsics
    intrinsics = load_json(os.path.join(base_dir, 'camera_intrinsics.json'))
    K = np.array(intrinsics['intrinsics'])
    # scale focal lengths & optical center
    K[0, 0] = K[0, 0] * scale
    K[1, 1] = K[1, 1] * scale
    K[0, 2] = K[0, 2] * scale
    K[1, 2] = K[1, 2] * scale
    
    # Get all camera directories in annots
    annot_dir = os.path.join(base_dir, 'annots')
    camera_dirs = [d for d in os.listdir(annot_dir) if os.path.isdir(os.path.join(annot_dir, d))]
    
    max_bbox_dim = 0
    bbox_info = {}
    
    # First pass: find the largest bbox dimension
    for camera in camera_dirs:
        camera_annot_dir = os.path.join(annot_dir, camera)

        annot_path = os.path.join(camera_annot_dir, f"{image_id}_img.json")
        annot = load_json(annot_path)
        
        bbox = get_bbox_from_annot(annot)
        (center_x, center_y), (width, height) = get_bbox_center_and_size(bbox)
        center_x = center_x * scale
        center_y = center_y * scale
        width = width * scale
        height = height * scale
        max_dim = max(width, height)
        max_bbox_dim = max(max_bbox_dim, max_dim)
        
        bbox_info[camera] = {
            'center': (center_x, center_y),
            'width': width,
            'height': height,
            'original_width': annot['width'] * scale,
            'original_height': annot['height'] * scale
        }

    results = {}
    
    # Second pass: compute crops and update intrinsics
    for camera, info in bbox_info.items():
        center_x, center_y = info['center']
        # these below are already scaled
        original_width = info['original_width']
        original_height = info['original_height']
        
        # Calculate crop coordinates
        half_size = max_bbox_dim // 2
        crop_x1 = int(center_x - half_size)
        crop_y1 = int(center_y - half_size)
        crop_x2 = crop_x1 + max_bbox_dim
        crop_y2 = crop_y1 + max_bbox_dim
        
        # Adjust crop if it goes outside image bounds
        if crop_x1 < 0:
            crop_x2 -= crop_x1
            crop_x1 = 0
        if crop_y1 < 0:
            crop_y2 -= crop_y1
            crop_y1 = 0
        if crop_x2 > original_width:
            crop_x1 -= (crop_x2 - original_width)
            crop_x2 = original_width
        if crop_y2 > original_height:
            crop_y1 -= (crop_y2 - original_height)
            crop_y2 = original_height
        

        # Update intrinsics for this camera
        K_new = update_intrinsics(K, crop_x1, crop_y1, original_width, original_height)
        
        results[camera] = {
            'crop': [crop_x1, crop_y1, crop_x2, crop_y2], # how much was cropped from the image (NOT intrinsics)
            'K': K_new.tolist(),
            'crop_size': max_bbox_dim
        }
    
    return results


def create_transforms_json(base_dir, output_dir, timestep,save=False):
    """Create transforms.json with cropped images and updated intrinsics."""
    # Load crop parameters
    crop_params = process_images_and_intrinsics(base_dir, timestep)
    
    # Load camera extrinsics
    extrinsics = load_json(os.path.join(base_dir, 'camera_extrinsics.json'))
    
    # Create output directory for cropped images if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare transforms.json structure
    transforms = {
        "frames": []
    }
    
    # Process each camera
    for camera_id, camera_params in crop_params.items():
        # Get camera extrinsics
        camera_data = extrinsics.get(f"1_{camera_id}.png")
        if not camera_data:
            print(f"Warning: No extrinsics found for camera {camera_id}. This indicates a BIG error.")
            continue
            
        # Get crop parameters
        crop = camera_params['crop'] # at this point, this is a square
        K = np.array(camera_params['K']) # 0.5 scaled intrinsics
        crop_size = camera_params['crop_size']
        
        # Create frame entry
        frame = {
            "file_path": f"{output_dir}/{camera_id}_cropped.png",
            "transform_matrix": [
                [float(x) for x in camera_data['rotation'][0]],
                [float(x) for x in camera_data['rotation'][1]],
                [float(x) for x in camera_data['rotation'][2]],
                [0.0, 0.0, 0.0, 1.0]
            ],
            "w": crop_size,
            "h": crop_size,
            "fl_x": float(K[0, 0]),
            "fl_y": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2])
            # downsample due to dataset downsampling
            # extrinsics and intrinsics are NOT downsampled by default in their jsons.
        }
        
        transforms["frames"].append(frame)
        
        # Load and crop image
        try:
            img_path = os.path.join(base_dir, 'images_lr', camera_id, f"{IMAGE_AT_TIME}_img.jpg")  # Assuming same timestep
            mask_path = os.path.join(base_dir, 'fmask_lr', camera_id, f"{IMAGE_AT_TIME}_img_fmask.png")
            if os.path.exists(img_path):
                img = np.array(Image.open(img_path))
                mask = np.array(Image.open(mask_path))
                img_masked = apply_mask(img, mask)
                img_masked_pil = Image.fromarray(img_masked)
                cropped = img_masked_pil.crop(crop)
                cropped.save(os.path.join(output_dir, f'{camera_id}_cropped_{IMAGE_AT_TIME}.png'))
            else:
                print(f"Warning: Image not found at {img_path}")
        except Exception as e:
            print(f"Error processing image for camera {camera_id}: {e}")
    
    # Save transforms.json
    if save:
        with open(os.path.join(output_dir, 'transforms.json'), 'w') as f:
            json.dump(transforms, f, indent=4)
        print(f"Created transforms.json with {len(transforms['frames'])} frames")

    return transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and crop multi-view images')
    parser.add_argument('--base_dir', type=str, default='.', required=True,
                      help='Base directory containing images and camera data. This should be where mvhumannet resides.')
    parser.add_argument('--timestep', type=int, default=5, required=True,
                      help='Timestep/frame to choose.')
    parser.add_argument('--output_dir', type=str, default='cropped_images', required=True,
                      help='Output directory for cropped images, updated transforms.json')
    
    args = parser.parse_args()
    
    # Update global variables based on arguments
    BASE_DIR = args.base_dir
    assert args.timestep % 5 == 0, "Timestep must be a multiple of 5 lower than the maximum timestep."
    IMAGE_AT_TIME = f"{str(args.timestep).zfill(4)}"

    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    transforms = create_transforms_json(BASE_DIR, args.output_dir, IMAGE_AT_TIME, save=True)
    transforms["camera_model"] = "OPENCV"
    with open(os.path.join(args.output_dir, 'transforms.json'), 'w') as f:
        json.dump(transforms, f, indent=4)
    print(f"Cropped images saved in {args.output_dir}/")