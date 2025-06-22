import numpy as np
# def estimate_depth(image):
#     # Dummy implementation (replace with real model inference)
#     h, w = image.shape[:2]
#     return np.full((h, w), 2.0, dtype=np.float32)  # Fake constant depth
def estimate_depth(image):
    h, w = image.shape[:2]
    # Make depth vary slightly with vertical position
    return np.tile(np.linspace(2.0, 3.0, h).reshape(h, 1), (1, w)).astype(np.float32)



