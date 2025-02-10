import os
import json
import cv2
import numpy as np

# --- Directories (adjust these as needed) ---
# For example, processing the training split:
images_dir = 'deepfashion2/images/val'
labels_dir = 'deepfashion2/labels/val'
masks_dir = 'deepfashion2/masks/val'

# Create the masks directory if it does not exist
os.makedirs(masks_dir, exist_ok=True)

# List through all JSON files in your labels directory
for label_filename in os.listdir(labels_dir):
    if not label_filename.endswith('.json'):
        continue

    label_path = os.path.join(labels_dir, label_filename)
    with open(label_path, 'r') as f:
        data = json.load(f)

    # Assume the image file has the same basename but with a .jpg extension.
    basename = os.path.splitext(label_filename)[0]
    image_path = os.path.join(images_dir, basename + '.jpg')
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image {image_path} not found. Skipping.")
        continue

    height, width = image.shape[:2]

    # Create an empty mask (single-channel) with background = 0.
    mask = np.zeros((height, width), dtype=np.uint8)

    # Process each key in the JSON that corresponds to an annotated item.
    # (In your JSON, keys like "item1" and "item2" contain segmentation info.)
    for key, item in data.items():
        if not key.startswith('item'):
            continue  # Skip keys like 'source' or 'pair_id'

        # Get the class id to paint the mask (adjust if you want to remap category_ids)
        category_id = item.get('category_id', 0)
        # Get the list of segmentation polygons
        segmentations = item.get('segmentation', [])
        for poly in segmentations:
            # Convert the flat list to a NumPy array of shape (-1, 2)
            pts = np.array(poly, np.int32).reshape((-1, 2))
            # Fill the polygon on the mask with the class id
            cv2.fillPoly(mask, [pts], color=category_id)

    # Save the mask image (use .png so that pixel values are preserved)
    mask_save_path = os.path.join(masks_dir, basename + '.png')
    cv2.imwrite(mask_save_path, mask)
    print(f"Processed {basename}")