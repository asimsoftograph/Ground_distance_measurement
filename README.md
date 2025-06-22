🖼️ Poster-to-Ground Distance Estimation (GUI Tool)

-> Detect posters or ads using a trained object detection model.

-> Segment ground areas using semantic segmentation.

-> Estimate pixel and real-world distance between the bottom of posters and the ground, using depth estimation.

-> Visualize results with bounding boxes, overlay masks, and distance annotations.




🤖 Models Used
-> poster_model.pt: YOLOv8 model trained on poster/ad dataset

-> ground_segmentation_model.pt: YOLOv8-seg 

-> estimate_depth(): Uses lightweight depth estimation model (e.g., MiDaS)
