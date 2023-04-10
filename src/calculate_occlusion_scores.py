import numpy as np
import os
import glob
from PIL import Image
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def resize_mask_nearest_neighbor(mask, target_width, target_height):
    return mask.resize((target_width, target_height), resample=Image.NEAREST)

def calculate_global_occlusion(occlusion_mask):
    # Define the occlusion classes and weights
    occlusion_classes = {
        'clean': {'rgb': (0, 0, 0), 'weight': 0},
        'transparent': {'rgb': (0, 255, 0), 'weight': 0.25},
        'semi-transparent': {'rgb': (0, 0, 255), 'weight': 0.5},
        'opaque': {'rgb': (255, 0, 0), 'weight': 1}
    }

    # Calculate the weighted global occlusion
    weighted_sum = 0
    total_pixels = occlusion_mask.shape[0] * occlusion_mask.shape[1]

    for occlusion_class in occlusion_classes.values():
        mask = np.all(occlusion_mask == occlusion_class['rgb'], axis=-1)
        num_occluded_pixels = np.count_nonzero(mask)
        weighted_sum += num_occluded_pixels * occlusion_class['weight']

    weighted_global_occlusion = weighted_sum / total_pixels

    return weighted_global_occlusion

def calculate_class_occlusion(occlusion_image_path, class_probability_map):
    # Load the occlusion image
    occlusion_image = Image.open(occlusion_image_path).convert('RGB')

    # Convert the occlusion image to a numpy array
    occlusion_array = np.array(occlusion_image)

    # Define the occlusion classes
    occlusion_classes = {
        'clean': {'rgb': (0, 0, 0), 'weight': 0},
        'transparent': {'rgb': (0, 255, 0), 'weight': 0.25},
        'semi-transparent': {'rgb': (0, 0, 255), 'weight': 0.5},
        'opaque': {'rgb': (255, 0, 0), 'weight': 1}
    }

    # Create a mask for the class of interest (probability > 0)
    class_mask = class_probability_map > 0

    # Calculate the class occlusion
    weighted_occluded_pixels = 0
    total_class_pixels = np.count_nonzero(class_mask)

    for occlusion_class in occlusion_classes.values():
        if occlusion_class['rgb'] != (0, 0, 0):  # Ignore the clean class
            mask = np.all(occlusion_array == occlusion_class['rgb'], axis=-1)
            occluded_class_pixels = np.count_nonzero(mask & class_mask)
            weighted_occluded_pixels += occluded_class_pixels * occlusion_class['weight']

    if total_class_pixels > 0:
        class_occlusion = weighted_occluded_pixels / total_class_pixels
    else:
        class_occlusion = 0

    return class_occlusion


def calculate_actual_occlusion(occlusion_mask, labeled_image, class_rgb_values):
    # Define the occlusion classes
    occlusion_classes = {
        'clean': {'rgb': (0, 0, 0), 'weight': 0},
        'transparent': {'rgb': (0, 255, 0), 'weight': 0.25},
        'semi-transparent': {'rgb': (0, 0, 255), 'weight': 0.5},
        'opaque': {'rgb': (255, 0, 0), 'weight': 1}
    }

    # Calculate the actual occlusion for each class label
    class_actual_occlusions = {}
    for class_label, class_rgb in class_rgb_values.items():
        class_mask = np.all(labeled_image == class_rgb, axis=-1)
        total_class_pixels = np.count_nonzero(class_mask)

        if total_class_pixels > 0:
            weighted_occluded_pixels = 0

            for occlusion_class in occlusion_classes.values():
                if occlusion_class['rgb'] != (0, 0, 0):  # Ignore the clean class
                    occlusion_mask_class = np.all(occlusion_mask == occlusion_class['rgb'], axis=-1)
                    masked_occlusion = np.logical_and(class_mask, occlusion_mask_class)
                    occluded_pixels = np.count_nonzero(masked_occlusion)
                    weighted_occluded_pixels += occluded_pixels * occlusion_class['weight']

            actual_occlusion = weighted_occluded_pixels / total_class_pixels
        else:
            actual_occlusion = 0

        class_actual_occlusions[class_label] = actual_occlusion

    return class_actual_occlusions

# Load probability maps
probability_maps = {}  # Load the probability maps you generated earlier

# Process occlusion images
occlusion_image_folder = "path/to/occlusion/image/folder"
occlusion_image_files = glob.glob(os.path.join(occlusion_image_folder, "*.png"))

results = []

for occlusion_image_file in occlusion_image_files:
    # Load corresponding labeled image
    labeled_image_file = "path/to/labeled/image"  # Replace this with the appropriate path to the labeled image
    labeled_image = Image.open(labeled_image_file).convert('RGB')
	width, height = labeled_image.size

	occlusion_mask = Image.open(occlusion_image_file).convert('RGB')
	resized_occlusion_mask = resize_mask_nearest_neighbor(occlusion_mask, width, height)

	occlusion_mask_np = np.array(resized_occlusion_mask)
	labeled_image_np = np.array(labeled_image)

    global_occlusion = calculate_global_occlusion(occlusion_mask_np)
    actual_occlusion = calculate_actual_occlusion(occlusion_mask_np, labeled_image_np, class_rgb_values)

    class_results = []

    for class_name, probability_map in probability_maps.items():
        class_occlusion = calculate_class_occlusion(occlusion_mask, probability_map)
        class_results.append((class_name, global_occlusion, class_occlusion, actual_occlusion))

    results.append(class_results)

# Evaluate the fit of the global estimate and the class ROI estimate as models
global_mse = mean_squared_error([x[1] for x in results], [x[3] for x in results])
class_roi_mse = mean_squared_error([x[2] for x in results], [x[3] for x in results])

print(f"Global Estimate Mean Squared Error: {global_mse}")
print(f"Class ROI Estimate Mean Squared Error: {class_roi_mse}")


# Assuming `results` is a list of tuples containing (class_name, global_occlusion, class_occlusion, actual_occlusion)

# Prepare data
global_occlusion_values = [x[1] for x in results]
class_occlusion_values = [x[2] for x in results]
actual_occlusion_values = [x[3] for x in results]

# Create scatter plot for global occlusion vs actual occlusion
plt.figure()
plt.scatter(global_occlusion_values, actual_occlusion_values, alpha=0.5)
plt.xlabel('Global Occlusion')
plt.ylabel('Actual Occlusion')
plt.title('Global Occlusion vs Actual Occlusion')
plt.grid(True)
plt.savefig('global_vs_actual_occlusion.png')

# Create scatter plot for class occlusion vs actual occlusion
plt.figure()
plt.scatter(class_occlusion_values, actual_occlusion_values, alpha=0.5)
plt.xlabel('Class Occlusion')
plt.ylabel('Actual Occlusion')
plt.title('Class Occlusion vs Actual Occlusion')
plt.grid(True)
plt.savefig('class_vs_actual_occlusion.png')

# Show the plots
plt.show()
