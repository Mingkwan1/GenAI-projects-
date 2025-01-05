import os
import cv2
from ultralytics import YOLO

# Step 1: Load YOLOv8 Model
model_path = "yolov8n.pt"  # Use pre-trained YOLOv8 nano model
model = YOLO(model_path)   # Load the YOLO model

# Step 2: Custom Configuration
# Define the custom dataset location
custom_dataset_yaml = "dataset.yaml"  # Replace with your dataset YAML path

# Check if a fine-tuned model exists; if not, fine-tune the YOLO model
fine_tuned_model_path = "runs/detect/train/weights/best.pt"
if not os.path.exists(fine_tuned_model_path):
    print("No fine-tuned model found. Training the model...")
    model.train(
        data=custom_dataset_yaml,
        epochs=50,
        imgsz=640,
        workers=4,
        name="quality_inspection"
    )

# Load the fine-tuned model
model = YOLO(fine_tuned_model_path)

# Step 3: Real-Time Defect Detection
def perform_inspection(input_source="video.mp4", output_path="output.avi"):
    """
    Perform defect detection on a video feed or camera.
    :param input_source: Path to the input video or integer for webcam.
    :param output_path: Path to save the annotated output video.
    """
    # Open the video source (0 for webcam, or video file path)
    cap = cv2.VideoCapture(input_source)

    # Define video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    print("Starting inspection...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform YOLO detection
        results = model(frame)
        annotated_frame = results[0].plot()  # Annotate the frame with detection boxes

        # Display the results
        cv2.imshow("Defect Detection", annotated_frame)

        # Write the frame to the output file
        out.write(annotated_frame)

        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Inspection completed. Results saved to {output_path}")

# Step 4: Run Inspection on a Video or Webcam
if __name__ == "__main__":
    # Replace 'video.mp4' with 0 for webcam or path to a test video
    perform_inspection(input_source="video.mp4", output_path="output.avi")
