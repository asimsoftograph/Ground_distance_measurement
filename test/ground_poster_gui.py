import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

# --- Load Models ---
ground_model = YOLO('F:/softograph/ground_distance/models/segment_ground_yolov8_nano.pt')  # your segmentation model
poster_model = YOLO('F:/softograph/ground_distance/models/new_grid_poster_m.pt')           # your poster detection model


class PosterGroundGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ground Segmentation + Poster Detection Viewer")

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
        original = image.copy()
        h, w = original.shape[:2]

        # --- Run ground segmentation ---
        ground_result = ground_model.predict(original, imgsz=640, conf=0.4, verbose=False)[0]
        mask_data = ground_result.masks.data[0].cpu().numpy() if ground_result.masks is not None else None

        if mask_data is not None:
            ground_mask = cv2.resize(mask_data.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            # Apply colored overlay for ground
            overlay = original.copy()
            overlay[ground_mask == 1] = (0, 255, 0)  # green for ground
            alpha = 0.5
            image = cv2.addWeighted(overlay, alpha, original, 1 - alpha, 0)

        # --- Run poster detection ---
        poster_result = poster_model.predict(original, imgsz=640, conf=0.4, verbose=False)[0]

        for box in poster_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red poster box

        # Convert to Tkinter image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(image_rgb)
        im_pil = im_pil.resize((800, 600))
        self.tk_img = ImageTk.PhotoImage(image=im_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

# --- Launch App ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PosterGroundGUI(root)
    root.mainloop()
