import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

# --- Load Models ---
ground_model = YOLO('F:/softograph/ground_distance/models/segment_ground_yolov8_nano.pt')
poster_model = YOLO('F:/softograph/ground_distance/models/new_grid_poster_m.pt')


def calculate_distance_from_ground(ground_mask, bbox):
    """
    Calculate pixel distance from top of ground to bottom of poster.
    """
    x1, y1, x2, y2 = bbox
    poster_bottom_y = y2

    cropped = ground_mask[:, x1:x2]
    ground_y_indices = np.where(cropped > 0)[0] #Finds where ground exists under that poster

    if ground_y_indices.size == 0:
        return None

    ground_top_y = np.min(ground_y_indices) # Topmost pixel of ground in the cropped area
    distance = poster_bottom_y - ground_top_y
    return distance


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

        # --- Ground Segmentation ---
        ground_result = ground_model.predict(original, imgsz=640, conf=0.4, verbose=False)[0]
        mask_data = ground_result.masks.data[0].cpu().numpy() if ground_result.masks is not None else None

        ground_mask = np.zeros((h, w), dtype=np.uint8)
        if mask_data is not None:
            ground_mask = cv2.resize(mask_data.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            overlay = original.copy()
            overlay[ground_mask == 1] = (0, 255, 0)  # Green for ground
            alpha = 0.5
            image = cv2.addWeighted(overlay, alpha, original, 1 - alpha, 0)

        # --- Poster Detection ---
        poster_result = poster_model.predict(original, imgsz=640, conf=0.4, verbose=False)[0]

        for box in poster_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Calculate pixel distance
            distance = calculate_distance_from_ground(ground_mask, (x1, y1, x2, y2))

            # Draw red box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Display distance
            if distance is not None:
                text = f"{distance} px above ground"
                cv2.putText(image, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- Display in Tkinter ---
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
