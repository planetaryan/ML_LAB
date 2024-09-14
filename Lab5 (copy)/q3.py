import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_image(path, size=(128, 64)):
    """Load a single image and resize it."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, size) if img is not None else None

def compute_hog(image, cell_size=8, block_size=2, bins=9):
    """Compute HoG descriptor for an image."""
    gx, gy = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1), cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hog_vector = []

    # Compute HoG for each cell
    for i in range(0, image.shape[0], cell_size):
        for j in range(0, image.shape[1], cell_size):
            cell_mag = magnitude[i:i + cell_size, j:j + cell_size]
            cell_ang = angle[i:i + cell_size, j:j + cell_size]
            hist, _ = np.histogram(cell_ang, bins=bins, range=(0, 180), weights=cell_mag)
            hog_vector.extend(hist)

    # Normalize over blocks
    block_stride = cell_size
    hog_descriptor = []
    for i in range(0, len(hog_vector) - block_size * bins + 1, block_stride * bins):
        block = hog_vector[i:i + block_size * bins]
        hog_descriptor.extend(block / (np.linalg.norm(block) + 1e-5))
    return np.array(hog_descriptor)

def sliding_window(image, window_size, step_size):
    """Generate sliding window patches."""
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def load_reference_images(directory, size=(128, 64)):
    """Load all reference images from a directory and compute their HoG descriptors."""
    hog_descriptors = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        img = load_image(path, size)
        if img is not None:
            hog_descriptor = compute_hog(img)
            hog_descriptors.append(hog_descriptor)
    return hog_descriptors

def detect_humans(image, ref_hogs, window_size=(128, 64), step_size=8, threshold=0.7):
    """Detect humans in an image using sliding window and multiple HoG descriptors."""
    detected_windows = []
    for (x, y, window) in sliding_window(image, window_size, step_size):
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            continue

        window_hog = compute_hog(window)
        scores = [cosine_similarity([window_hog], [ref_hog])[0, 0] for ref_hog in ref_hogs]
        max_score = max(scores) if scores else 0

        if max_score > threshold:
            detected_windows.append((x, y, max_score))

    return non_max_suppression(detected_windows)

def non_max_suppression(windows, overlap_threshold=0.5):
    """Suppress overlapping windows based on their scores."""
    if len(windows) == 0:
        return []

    boxes = np.array([[x, y, x + 128, y + 64] for (x, y, _) in windows])
    scores = np.array([score for (_, _, score) in windows])
    indices = np.argsort(scores)[::-1]

    selected = []
    while len(indices) > 0:
        i = indices[0]
        selected.append(windows[i])
        remaining = []

        for j in range(1, len(indices)):
            xx1 = max(boxes[i][0], boxes[indices[j]][0])
            yy1 = max(boxes[i][1], boxes[indices[j]][1])
            xx2 = min(boxes[i][2], boxes[indices[j]][2])
            yy2 = min(boxes[i][3], boxes[indices[j]][3])
            overlap = max(0, xx2 - xx1) * max(0, yy2 - yy1) / (
                (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) +
                (boxes[indices[j]][2] - boxes[indices[j]][0]) * (boxes[indices[j]][3] - boxes[indices[j]][1])
                - max(0, xx2 - xx1) * max(0, yy2 - yy1))

            if overlap <= overlap_threshold:
                remaining.append(indices[j])

        indices = np.array(remaining)

    return selected

# Load reference images and compute HoG descriptors
ref_hogs = load_reference_images('pos/')  # Ensure 'pos/' directory exists and contains images

# Load test image
test_image = load_image('images/humans.png', size=None)

# Detect humans in the test image
detected_windows = detect_humans(test_image, ref_hogs)

# Save the results to a new image
for (x, y, score) in detected_windows:
    cv2.rectangle(test_image, (x, y), (x + 128, y + 64), (255, 0, 0), 2)
cv2.imwrite('detected_humans.jpg', test_image)
