import numpy as np

# ITU-R BT.601 luma coefficients for R, G, B.
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])


def color_to_grayscale(image):
    """
    Convert an RGB (or RGBA) image of shape (H, W, C) to grayscale using
    the BT.601 luma formula:

        gray = 0.299 * R + 0.587 * G + 0.114 * B

    Any alpha channel is ignored. Returns a nested list of shape (H, W).
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"expected shape (H, W, 3) or (H, W, 4), got {image.shape}")
    return (image[:, :, :3] @ LUMA_WEIGHTS).tolist()
