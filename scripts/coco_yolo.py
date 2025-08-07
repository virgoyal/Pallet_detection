import os
import json

def coco_to_yolo_seg(json_path, images_dir, output_labels_dir):
    os.makedirs(output_labels_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Only keep class 2 and 3 (re-index to 0 and 1)
    keep_classes = {2: 0, 3: 1}

    # Map image ID to (file_name, width, height)
    images = {img['id']: (img['file_name'], img['width'], img['height']) for img in data['images']}
    labels = {img_id: [] for img_id in images.keys()}

    for ann in data['annotations']:
        cat_id = ann['category_id']
        if cat_id not in keep_classes:
            continue  # skip unwanted classes

        img_id = ann['image_id']
        new_class_id = keep_classes[cat_id]

        segmentation = ann.get('segmentation')
        if not segmentation or not isinstance(segmentation, list) or not segmentation[0]:
            continue  # skip if segmentation is empty or invalid

        file_name, img_width, img_height = images[img_id]
        coords = segmentation[0]  # assuming single polygon format
        norm_coords = [
            str(round(x / img_width, 6)) if i % 2 == 0 else str(round(x / img_height, 6))
            for i, x in enumerate(coords)
        ]
        labels[img_id].append(f"{new_class_id} " + ' '.join(norm_coords))

    for img_id, label_lines in labels.items():
        if not label_lines:
            continue
        file_stem = os.path.splitext(images[img_id][0])[0]
        label_path = os.path.join(output_labels_dir, f"{file_stem}.txt")
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))


if __name__ == "__main__":
    coco_to_yolo_seg(
        json_path="/home/akxhay/Desktop/coco/train/_annotations.coco.json",
        images_dir="/home/akxhay/Desktop/coco/train",
        output_labels_dir="/home/akxhay/Desktop/coco/train/labels"
    )
