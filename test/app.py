import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

# --- Load Models Once ---
ground_model = YOLO('F:/softograph/ground_distance/model/segment_ground_yolov8_nano.pt')  # your segmentation model
poster_model = YOLO('F:/softograph/ground_distance/model/new_grid_poster_m.pt')           # your poster detection model

# --- GUI App ---
class GroundToPosterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ground-to-Poster Distance App")

        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        self.image = cv2.imread(file_path)
        self.process_and_display(self.image)

    def process_and_display(self, image):
        h, w = image.shape[:2]

        # --- Run Ground Segmentation ---
        ground_result = ground_model.predict(image, imgsz=640, conf=0.4, verbose=False)[0]
        ground_mask = ground_result.masks.data[0].cpu().numpy()
        ground_mask = cv2.resize(ground_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        # --- Run Poster Detection ---
        poster_result = poster_model.predict(image, imgsz=640, conf=0.4, verbose=False)[0]

        for box in poster_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            poster_bottom_y = y2

            crop_ground = ground_mask[:, x1:x2]
            if np.count_nonzero(crop_ground) == 0:
                continue

            y_indices = np.where(crop_ground > 0)[0]
            if y_indices.size == 0:
                continue

            ground_top_y = np.min(y_indices)
            pixel_distance = poster_bottom_y - ground_top_y

            # Draw results
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.line(image, (x1, ground_top_y), (x2, ground_top_y), (0, 255, 0), 2)
            cv2.putText(image, f'{int(pixel_distance)}px', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- Show in Tkinter ---
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(image_rgb)
        im_pil = im_pil.resize((800, 600))
        self.tk_img = ImageTk.PhotoImage(image=im_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

# --- Run App ---
if __name__ == "__main__":
    root = tk.Tk()
    app = GroundToPosterApp(root)
    root.mainloop()
