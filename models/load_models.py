from ultralytics import YOLO

poster_model = YOLO("models/new_grid_poster_m.pt")
ground_model = YOLO("models/segment_ground_yolov8_nano.pt")
