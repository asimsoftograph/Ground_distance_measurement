import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from ultralytics import YOLO

# Load models
ground_model = YOLO('F:/softograph/ground_distance/models/segment_ground_yolov8_nano.pt')
poster_model = YOLO('F:/softograph/ground_distance/models/new_grid_poster_m.pt')

# Load MiDaS model for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

def estimate_depth(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    return depth_map

class PosterGroundApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Poster-Ground Distance with Depth")
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()
        self.btn = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.btn.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        img = cv2.imread(file_path)
        self.original = img.copy()
        self.process_and_display(img)
    def process_and_display(self, image):
      h, w = image.shape[:2]

    # Ground Segmentation
      ground_result = ground_model.predict(image, imgsz=640, conf=0.4, verbose=False)[0]
      ground_mask = None
      if ground_result.masks is not None:
        mask_data = ground_result.masks.data[0].cpu().numpy()
        ground_mask = cv2.resize(mask_data.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        ground_overlay = np.zeros_like(image, dtype=np.uint8)
        ground_overlay[ground_mask == 1] = (0, 255, 0)  # Green
        alpha = 0.4
        image = cv2.addWeighted(ground_overlay, alpha, image, 1 - alpha, 0)

    # Depth Estimation
      depth_map = estimate_depth(self.original)
      depth_map = cv2.resize(depth_map, (w, h))  # Ensure same size as image

    # Poster Detection
      poster_result = poster_model.predict(self.original, imgsz=640, conf=0.4, verbose=False)[0]

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

    # Show in GUI
      img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      pil_img = Image.fromarray(img_rgb).resize((800, 600))
      self.tk_img = ImageTk.PhotoImage(image=pil_img)
      self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)


# Launch GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = PosterGroundApp(root)
    root.mainloop()