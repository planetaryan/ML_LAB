import cv2
import numpy as np
import matplotlib.pyplot as plt

# Harris Corner Detection
def harris_corner_detection(image, k=0.04, threshold=0.01):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Convert to float32
    gray = np.float32(gray)

    # Compute gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    Ixx = sobel_x**2
    Iyy = sobel_y**2
    Ixy = sobel_x * sobel_y

    # Compute Harris response
    height, width = gray.shape
    R = np.zeros((height, width), dtype=np.float64)

    for i in range(1, height-1):
        for j in range(1, width-1):
            Sxx = np.sum(Ixx[i-1:i+2, j-1:j+2])
            Syy = np.sum(Iyy[i-1:i+2, j-1:j+2])
            Sxy = np.sum(Ixy[i-1:i+2, j-1:j+2])

            det_M = Sxx * Syy - Sxy**2
            trace_M = Sxx + Syy
            R[i, j] = det_M - k * trace_M**2

    # Threshold on R to detect corners
    corners = R > (threshold * R.max())
    result = image.copy()
    result[corners] = [0, 0, 255]  # Red corners on the image

    return result

# FAST Corner Detection
def fast_corner_detection(image, threshold=20):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray = np.float32(gray)
    height, width = gray.shape
    result = image.copy()

    def is_corner(x, y):
        p = gray[x, y]
        circle = [(x + dx, y + dy) for dx in range(-3, 4) for dy in range(-3, 4) if (dx**2 + dy**2 <= 3**2) and (dx != 0 or dy != 0)]
        circle_values = [gray[x+dx, y+dy] for dx, dy in circle if 0 <= x+dx < height and 0 <= y+dy < width]
        if len(circle_values) < 16:
            return False

        num_strong = sum(abs(c - p) > threshold for c in circle_values)
        return num_strong >= 12

    for i in range(3, height-3):
        for j in range(3, width-3):
            if is_corner(i, j):
                cv2.circle(result, (j, i), 3, (255, 0, 0), -1)  # Draw a blue circle

    return result

def main(image_path):
    # Read image
    image = cv2.imread(image_path)

    # Harris Corner Detection
    harris_result = harris_corner_detection(image, k=0.04, threshold=0.01)

    # FAST Corner Detection
    fast_result = fast_corner_detection(image, threshold=20)

    # Plot results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(harris_result, cv2.COLOR_BGR2RGB))
    plt.title('Harris Corners Detected')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(fast_result, cv2.COLOR_BGR2RGB))
    plt.title('FAST Keypoints Detected')

    plt.show()

# Example usage
if __name__ == "__main__":
    main('images/bricks.jpg')

