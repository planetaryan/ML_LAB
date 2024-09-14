import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, correlate
from sklearn.metrics import pairwise_distances

def gaussian_pyramid(image, levels):
    pyramid = [image]
    for i in range(1, levels):
        sigma = 1.6 * (2 ** (i - 1))
        blurred = gaussian_filter(image, sigma=sigma)
        pyramid.append(blurred)
    return pyramid

def difference_of_gaussian(pyramid):
    dog_pyramid = []
    for i in range(len(pyramid) - 1):
        dog = pyramid[i + 1] - pyramid[i]
        dog_pyramid.append(dog)
    return dog_pyramid

def detect_keypoints(dog_pyramid):
    keypoints = []
    for i, dog in enumerate(dog_pyramid):
        local_max = (dog == correlate(dog, np.ones((3, 3, 3)), mode='constant'))
        keypoints.append(local_max)
    return keypoints

def compute_descriptors(image, keypoints):
    descriptors = []
    for kp in keypoints:
        descriptors.append(np.random.rand(128))
    return np.array(descriptors)

def apply_transformations(image):
    rows, cols = image.shape
    M_scale = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 0.5)
    transformed_image = cv2.warpAffine(image, M_scale, (cols, rows))
    return transformed_image

def compare_descriptors(desc1, desc2):
    distances = pairwise_distances(desc1, desc2, metric='euclidean')
    return distances

def custom_sift(image):
    pyramid = gaussian_pyramid(image, levels=5)
    dog_pyramid = difference_of_gaussian(pyramid)
    keypoints = detect_keypoints(dog_pyramid)
    descriptors = compute_descriptors(image, keypoints)
    return keypoints, descriptors

def opencv_sift(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def compare_sift_implementations(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    custom_keypoints, custom_descriptors = custom_sift(image)
    opencv_keypoints, opencv_descriptors = opencv_sift(image)
    transformed_image = apply_transformations(image)
    _, transformed_descriptors = opencv_sift(transformed_image)
    distances = compare_descriptors(opencv_descriptors, transformed_descriptors)
    print(f"Descriptor distance matrix shape: {distances.shape}")
    img_keypoints = cv2.drawKeypoints(image, opencv_keypoints, None)
    cv2.imshow('Keypoints', img_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

compare_sift_implementations('image.jpg')
