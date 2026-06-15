import numpy as np

def color_to_grayscale(image):
    image = np.array(image)
    return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).tolist()
