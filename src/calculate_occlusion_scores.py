import numpy as np
import os
import glob
from PIL import Image
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import csv

def resize_mask_nearest_neighbor(mask, target_width, target_height):
    return mask.resize((target_width, target_height), resample=Image.NEAREST)

def load_probability_map_from_csv(file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        data = [list(map(float, row)) for row in csv_reader]
    return np.array(data)

def write_mse_to_csv(filename, mse_results):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Class', 'MSE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for class_name, mse in mse_results.items():
            writer.writerow({'Class': class_name, 'MSE': mse})


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

def calculate_class_occlusion(occlusion_array, class_probability_map):
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

class_probability_map_files = {
    #"cassed_mavs_forest": "CaSSed MAVS_forest_map.csv",
    #"cassed_mavs_road": "CaSSed MAVS_road_map.csv",
    #"cassed_mavs_veg": "CaSSed MAVS_vegetation_map.csv",
    #"city_road": "CityScapes gtFine_road_map.csv",
    #"city_forest": "CityScapes gtFine_vegetation_map.csv",
    #"city_person": "CityScapes gtFine_person_map.csv",
    #"water": "Rellis3D_water_map.csv",
    "vehicle": "Rellis3D_vehicle_map.csv",
    "tree": "Rellis3D_tree_map.csv",
    "sky": "Rellis3D_sky_map.csv",
    #"rubble": "Rellis3D_rubble_map.csv",
    "puddle": "Rellis3D_puddle_map.csv",
    #"pole": "Rellis3D_pole_map.csv",
    "person": "Rellis3D_person_map.csv",
    "object": "Rellis3D_object_map.csv",
    "mud": "Rellis3D_mud_map.csv",
    #"log": "Rellis3D_log_map.csv",
    "grass": "Rellis3D_grass_map.csv",
}

class_rgb_values = {
    'grass': (0,102,0),
    'tree': (0,255,0),
    #'pole': (0,153,153),
    #'water': (0,128,255),
    'sky': (0,0,255),
    'vehicle': (255,255,0),
    'object': (255,0,127),
    #'log': (102,0,0),
    'person': (204,153,255),
    'puddle': (134,255,239),
    'mud': (99,66,34),
    #'rubble': (110,22,138),
}

# Load probability maps
# Load probability maps
probability_maps = {}
for class_name, map_file in class_probability_map_files.items():
    probability_map = load_probability_map_from_csv(map_file)
    probability_maps[class_name] = probability_map
    print(f"Dimensions of {class_name} probability map: {probability_map.shape}")

classwise_data = {}
for class_name in class_probability_map_files.keys():
    classwise_data[class_name] = {"actual": [], "predicted": []}


# Process occlusion images
occlusion_image_folder = "occlusion_imgs\\"
occlusion_image_files = glob.glob(os.path.join(occlusion_image_folder, "*_FV.png"))

results = []

for occlusion_image_file in occlusion_image_files:
    # Load corresponding labeled image
    labeled_image_file = "I:\\projects\\ARC\\Project1.38\\datasets\\rellis3d\\Rellis_3D_pylon_camera_node_label_color\\Rellis-3D\\00000\\pylon_camera_node_label_color\\frame000106-1581624663_349.png"  # Replace this with the appropriate path to the labeled image
    labeled_image = Image.open(labeled_image_file).convert('RGB')
    width, height = labeled_image.size
    

    occlusion_mask = Image.open(occlusion_image_file).convert('RGB')
    resized_occlusion_mask = resize_mask_nearest_neighbor(occlusion_mask, width, height)

    occlusion_mask_np = np.array(resized_occlusion_mask)
    labeled_image_np = np.array(labeled_image)
    
    class_counts = {}
    for class_name, rgb_value in class_rgb_values.items():
        class_mask = (labeled_image_np == rgb_value).all(axis=-1)
        class_counts[class_name] = np.count_nonzero(class_mask)

    print(f"Counts of each class: {class_counts}")
    print(f"Dimensions of labeled image: {labeled_image_np.shape[:-1]}")
    print(f"occlusion_mask_np shape: {occlusion_mask_np.shape}")
    print(f"probability_map shape: {probability_map.shape}")

    global_occlusion = calculate_global_occlusion(occlusion_mask_np)
    actual_occlusion = calculate_actual_occlusion(occlusion_mask_np, labeled_image_np, class_rgb_values)

    for class_name, probability_map in probability_maps.items():
        class_occlusion = calculate_class_occlusion(occlusion_mask_np, probability_map)
        results.append((class_name, global_occlusion, class_occlusion, actual_occlusion[class_name]))
        if class_name not in classwise_data:
            classwise_data[class_name] = {"actual": [], "predicted": []}

        classwise_data[class_name]["actual"].append(actual_occlusion[class_name])
        classwise_data[class_name]["predicted"].append(class_occlusion)

# Evaluate the fit of the global estimate and the class ROI estimate as models
global_mse = mean_squared_error([x[1] for x in results], [x[3] for x in results])
#class_roi_mse = mean_squared_error([x[2] for x in results], [x[3] for x in results])
print(classwise_data)
mse_results = {}
for class_name, data in classwise_data.items():
    mse = mean_squared_error(data["actual"], data["predicted"])
    mse_results[class_name] = mse

print(results)
print(f"Global Estimate Mean Squared Error: {global_mse}")
write_mse_to_csv('mse.csv', mse_results)
#print(f"Class ROI Estimate Mean Squared Error: {class_roi_mse}")


# Prepare data
class_labels = list(mse_results.keys())
mse_values = list(mse_results.values())
x = np.arange(len(class_labels) + 1)  # Add an extra position for the global MSE

# Create a bar plot
fig, ax = plt.subplots()
bar_width = 0.4
bars_classwise = ax.bar(x[:-1], mse_values, width=bar_width, label="Class-wise MSE")
bars_global = ax.bar(x[-1], global_mse, width=bar_width, label="Global MSE")

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel("Classes")
ax.set_ylabel("MSE")
ax.set_title("MSE for Each Class and Global Occlusion")
ax.set_xticks(x)
ax.set_xticklabels(class_labels + ["Global"])
ax.legend()

# Show the plot
plt.show()

class_rgb_normalized = {class_name: tuple(c / 255 for c in rgb) for class_name, rgb in class_rgb_values.items()}


# Assuming `results` is a list of tuples containing (class_name, global_occlusion, class_occlusion, actual_occlusion)
# Create scatter plots for global occlusion vs actual occlusion and class occlusion vs actual occlusion
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot global occlusion vs actual occlusion
for class_name, global_occlusion, class_occlusion, actual_occlusion in results:
    ax1.scatter(global_occlusion, actual_occlusion, alpha=0.5, color=class_rgb_normalized[class_name])

ax1.set_xlabel("Global Occlusion")
ax1.set_ylabel("Actual Occlusion")
ax1.set_title("Global Occlusion vs Actual Occlusion")
ax1.grid(True)

# Plot class occlusion vs actual occlusion
for class_name, global_occlusion, class_occlusion, actual_occlusion in results:
    ax2.scatter(class_occlusion, actual_occlusion, alpha=0.5, color=class_rgb_normalized[class_name])

ax2.set_xlabel("Class Occlusion")
ax2.set_ylabel("Actual Occlusion")
ax2.set_title("Class Occlusion vs Actual Occlusion")
ax2.grid(True)

# Create a custom legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=class_name, markerfacecolor=color, markersize=8)
                   for class_name, color in class_rgb_normalized.items()]
ax2.legend(handles=legend_elements, title="Classes", loc="upper left", bbox_to_anchor=(1,1))

# Show the plots
plt.show()


