import os
import cv2
import numpy as np
import supervision as sv
from roboflow import Roboflow
import pyresearch

# Constants
PROJECT_NAME = "elephant-detection-cxnt1"
VIDEO_FILE = "2835528-hd_1920_1080_25fps.mp4"
ANNOTATED_VIDEO = "output.mp4"


                
                                
     
                
# Initialize Roboflow API
rf = Roboflow(api_key="NqjtCN1BkDxTg2u1jXzs")
project = rf.workspace().project(PROJECT_NAME)
model = project.version(4).model

# Get video predictions
job_id, signed_url, expire_time = model.predict_video(
   VIDEO_FILE,
   fps=5,
   prediction_type="batch-video",
)
results = model.poll_until_video_results(job_id)

# Annotators for enhanced visual appeal
mask_annotator = sv.MaskAnnotator(opacity=0.6)  # Semi-transparent colored mask
box_annotator = sv.BoundingBoxAnnotator(thickness=2)  # Bounding boxes
label_annotator = sv.LabelAnnotator(text_scale=1.2, text_thickness=2)  # Clearer labels
tracker = sv.ByteTrack()  # Tracks objects frame-to-frame

# Open video
cap = cv2.VideoCapture(VIDEO_FILE)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video Writer to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(ANNOTATED_VIDEO, fourcc, frame_rate, (frame_width, frame_height))

# Function to generate unique mask colors for each class
def get_mask_color(class_name):
    np.random.seed(hash(class_name) % 256)
    return tuple(np.random.randint(50, 200, 3).tolist())  # Softer colors

# Function to annotate each frame with a **more attractive shaded effect**
def annotate_frame(frame: np.ndarray, frame_number: int) -> np.ndarray:
    try:
        time_offset = frame_number / frame_rate

        if "time_offset" in results and results["time_offset"]:
            closest_time_offset = min(results['time_offset'], key=lambda t: abs(t - time_offset))
            index = results['time_offset'].index(closest_time_offset)
            detection_data = results[PROJECT_NAME][index]

            roboflow_format = {
                "predictions": detection_data['predictions'],
                "image": {"width": frame.shape[1], "height": frame.shape[0]}
            }
            detections = sv.Detections.from_inference(roboflow_format)
            detections = tracker.update_with_detections(detections)

            # Ensure labels match the number of detections
            labels = [pred['class'] for pred in detection_data['predictions'][:len(detections)]]

        else:
            raise KeyError("Missing 'time_offset' in results")

    except (IndexError, KeyError, ValueError) as e:
        print(f"Exception in processing frame {frame_number}: {e}")
        detections = sv.Detections(
            xyxy=np.empty((0, 4)),
            confidence=np.empty(0),
            class_id=np.empty(0, dtype=int)
        )
        labels = []

    # **Apply advanced shaded mask effect**
    annotated_frame = frame.copy()
    for i, detection in enumerate(detections.xyxy):
        x1, y1, x2, y2 = map(int, detection)
        mask_color = get_mask_color(labels[i])  # Unique color for each object

        # Create a soft transparent overlay
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), mask_color, -1)
        cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)

    # **Apply bounding boxes and labels for clarity**
    annotated_frame = box_annotator.annotate(annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)

    return annotated_frame

# Process video with attractive shading
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if no frame is read

    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    annotated_frame = annotate_frame(frame, frame_number)

    # Save the frame to the output video
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()

print("✅ Video processing complete. Output saved as:", ANNOTATED_VIDEO)
