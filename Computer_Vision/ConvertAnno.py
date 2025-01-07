import os
import cv2

def create_yolo_annotations(ground_truth_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for mask_file in os.listdir(ground_truth_dir):
        mask_path = os.path.join(ground_truth_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        txt_path = os.path.join(save_dir, mask_file.replace('.png', '.txt'))
        with open(txt_path, 'w') as f:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_center = (x + w / 2) / mask.shape[1]
                y_center = (y + h / 2) / mask.shape[0]
                width = w / mask.shape[1]
                height = h / mask.shape[0]
                f.write(f"0 {x_center} {y_center} {width} {height}\n")

# Example Usage
ground_truth_dir = "D:\DumbStuff\GenAI-projects-\Computer_Vision\data\Labels\val\ground_truth"  # Ground truth folder
save_dir = "D:\DumbStuff\GenAI-projects-\Computer_Vision\data\Labels\val"
create_yolo_annotations(ground_truth_dir, save_dir)
