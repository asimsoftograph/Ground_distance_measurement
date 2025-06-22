import cv2
import numpy as np
from models.load_models import ground_model, poster_model
from depth.depth_model import estimate_depth

def detect_posters_and_ground(image, original):
    h, w = image.shape[:2]

    # Ground Segmentation
    ground_result = ground_model.predict(image, imgsz=640, conf=0.4, verbose=False)[0]
    ground_mask = None
    if ground_result.masks is not None:
        mask_data = ground_result.masks.data[0].cpu().numpy()
        ground_mask = cv2.resize(mask_data.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        ground_overlay = np.zeros_like(image, dtype=np.uint8)
        ground_overlay[ground_mask == 1] = (0, 255, 0)
        image = cv2.addWeighted(ground_overlay, 0.4, image, 0.6, 0)

    # Depth Estimation
    depth_map = estimate_depth(original)
    depth_map = cv2.resize(depth_map, (w, h))

    # Poster Detection
    poster_result = poster_model.predict(original, imgsz=640, conf=0.4, verbose=False)[0]

    # for box in poster_result.boxes:
    #     x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #     for label, (px, py) in [("L", (x1, y2)), ("R", (x2, y2))]:
    #         if ground_mask is None or py >= h or px >= w:
    #             continue

    #         try:
    #             poster_depth = depth_map[py, px]
    #         except:
    #             poster_depth = 0

    #         ground_y = None
    #         original_px = px
    #         window = 5

    #         for y in range(py, h):
    #             for dx in range(-window, window + 1):
    #                 nx = px + dx
    #                 if 0 <= nx < w and ground_mask[y, nx] == 1:
    #                     ground_y = y
    #                     px = nx
    #                     break
    #             if ground_y is not None:
    #                 break

    #         if ground_y is not None:
    #             try:
    #                 ground_depth = depth_map[ground_y, px]
    #                 pixel_dist = ground_y - py
    #                 real_dist = abs(ground_depth - poster_depth)
    #             except:
    #                 pixel_dist = 0
    #                 real_dist = 0
 
    #             cv2.line(image, (original_px, py), (px, ground_y), (255, 255, 0), 2)
            
    #             cv2.putText(image, f"PxDist: {pixel_dist}px", (original_px, py + 20),
    #                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    #             cv2.putText(image, f"Real_Dist : {real_dist:.2f}m", (original_px, py + 40),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
              
    #         else:
    #             cv2.putText(image, f"{label}: Ground Not Found", (px, py + 20),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # return image

    for box in poster_result.boxes:
         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

         bottom_left = (x1, y2)
         bottom_right = (x2, y2)

         for label, point in [("L", bottom_left), ("R", bottom_right)]:
            px, py = point    #  px and py are the pixel coordinates of the poster's bottom left and right corners

            if ground_mask is None or py >= h or px >= w:
                continue  # Skip invalid cases

            try:
                poster_depth = depth_map[py, px]
            except:
                poster_depth = 0

            ground_y = None
            original_px = px
            window = 5  # Search ±5 pixels horizontally

            for y in range(py, h):
                for dx in range(-window, window + 1):
                    nx = px + dx
                    if 0 <= nx < w and ground_mask[y, nx] == 1:
                        ground_y = y
                        px = nx  # Update px to ground pixel's x
                        break
                if ground_y is not None:
                    break

            if ground_y is not None:
                try:
                    ground_depth = depth_map[ground_y, px]
                    pixel_dist = ground_y - py
                    real_dist = abs(ground_depth - poster_depth)
                except:
                    pixel_dist = 0
                    real_dist = 0

                cv2.line(image, point, (px, ground_y), (255, 255, 0), 2)
                cv2.putText(image, f"PxDist: {pixel_dist}px", (original_px, py + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(image, f"Real_Dist : {real_dist:.2f}m", (original_px, py + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(image, f"{label}: Ground Not Found", (px, py + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)



    return image
