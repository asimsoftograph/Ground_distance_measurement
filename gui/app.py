import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.post_process import detect_posters_and_ground


class PosterGroundApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Poster to Ground Distance Calculator")
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack()

        self.tk_img = None
        self.original = None

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            self.original = image.copy()
            processed = detect_posters_and_ground(image, self.original)
            self.display_image(processed)

    def display_image(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize((800, 600))
        self.tk_img = ImageTk.PhotoImage(image=pil_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = PosterGroundApp(root)
    root.mainloop()
