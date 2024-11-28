import cv2
from ultralytics import YOLO
from utils.Functions.Stream import process_stream

# Streaming source 
cap1 = cv2.VideoCapture(0) # 0 for default camera, 1 for second camera
cap2 = cv2.VideoCapture(r"Test_Resources\Video\test_video.mp4") # path to video file

# Load the YOLO models
model_v8 = YOLO("yolov8n.pt")
model_v8_copy = YOLO("yolov8n.pt") 
#model_v9 = YOLO("yolov9n.pt")
#model_v10 = YOLO("yolov10n.pt")
#model_v11 = YOLO("yolov11n.pt")

# Process streams in sync
process_stream(cap1, cap2, model_v8, model_v8_copy) # each stream needs its own model
