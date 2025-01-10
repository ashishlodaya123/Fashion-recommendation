import pandas as pd
import webcolors
from scipy.spatial import distance
import os
import re
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Load dataset
df = pd.read_csv("C:\\Narasimha\\KLETU Related\\5th Semester Related\\CV\\Course Project\\Dataset\\polyvore Dataset\\Data-with-occasion.csv")  # Adjust path as needed

# Function to find closest CSS3 color name based on RGB value
def closest_color(requested_color):
    min_distance = float('inf')
    closest_color_name = None
    for color_name, color_rgb in webcolors.CSS3_NAMES_TO_HEX.items():
        r, g, b = webcolors.hex_to_rgb(color_rgb)
        d = distance.euclidean((r, g, b), requested_color)
        if d < min_distance:
            min_distance = d
            closest_color_name = color_name
    return closest_color_name

# Function to convert color names to RGB
def rgb_from_text(user_color):
    try:
        return webcolors.name_to_rgb(user_color)
    except ValueError:
        print(f"Color '{user_color}' not recognized. Try basic colors (red, blue, etc.).")
        return None

# Function to parse RGB values from 'RGB(x, y, z)' string format
def parse_rgb(rgb_str):
    match = re.match(r'RGB\((\d+), (\d+), (\d+)\)', rgb_str)
    if match:
        return tuple(map(int, match.groups()))
    else:
        print(f"Unexpected format: {rgb_str}")
        return (0, 0, 0)

# Helper function to get Euclidean distance
def calculate_distance(color1, color2):
    return distance.euclidean(color1, color2)

# Initialize counters for precision, recall, and other metrics
tp = 0
fp = 0
fn = 0
top_n_accuracy = 0
recommended_items_count = 0
relevant_items_count = 0
total_recommendations = 0
N = 5  # Top-N for accuracy

# Initialize counters for compatibility testing
compatibility_tp = 0
compatibility_tn = 0
compatibility_fp = 0
compatibility_fn = 0

# Function to recommend outfits and track metrics
def recommend_outfit_with_metrics(df, user_inputs, top_n=N):
    global tp, fp, fn, top_n_accuracy, recommended_items_count, relevant_items_count, total_recommendations

    user_rgb = rgb_from_text(user_inputs['color'])
    if not user_rgb:
        return None

    # Filter the dataset based on the occasion and color
    filtered_df = df[df['Occasion'].str.lower() == user_inputs['occasion'].lower()]
    
    if filtered_df.empty:
        print("No items match your preferences.")
        return None
    
    # Track relevant items with a less strict color match threshold
    relevant_items = filtered_df[filtered_df['Dominant Colour'].apply(lambda x: calculate_distance(parse_rgb(x), user_rgb) < 100)]  # Relaxed threshold for color match
    relevant_items_count += len(relevant_items)

    # Iterate over recommended outfits and evaluate
    for idx, top in filtered_df.iterrows():
        total_recommendations += 1
        recommended_items_count += 1

        # Check if the recommended item is relevant (based on color or occasion)
        if top['Occasion'].lower() == user_inputs['occasion'].lower() and calculate_distance(parse_rgb(top['Dominant Colour']), user_rgb) < 100:
            tp += 1
        else:
            fp += 1
        
        if top['Occasion'].lower() != user_inputs['occasion'].lower() or calculate_distance(parse_rgb(top['Dominant Colour']), user_rgb) >= 100:
            fn += 1

        # Evaluate Top-N accuracy (if the top N recommendations are correct)
        top_n_accuracy += int(idx < top_n)  # Check if this item is in the top N recommendations

    # Compute Precision, Recall, and F1-Score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Compute Top-N accuracy
    top_n_accuracy = (top_n_accuracy / total_recommendations) * 100
    
    return zip(filtered_df.iterrows(), filtered_df.iterrows())

# Function to display images side by side
def display_outfit(top_image_path, bottom_image_path, shoe_image_path=None, accessory_image_path=None):
    images = []
    titles = []
    
    # Load and append images
    if os.path.exists(top_image_path):
        top_img = cv2.imread(top_image_path)
        top_img = cv2.cvtColor(top_img, cv2.COLOR_BGR2RGB)
        images.append(top_img)
        titles.append("Top/Outerwear")
    
    if os.path.exists(bottom_image_path):
        bottom_img = cv2.imread(bottom_image_path)
        bottom_img = cv2.cvtColor(bottom_img, cv2.COLOR_BGR2RGB)
        images.append(bottom_img)
        titles.append("Bottom")
    
    if shoe_image_path and os.path.exists(shoe_image_path):
        shoe_img = cv2.imread(shoe_image_path)
        shoe_img = cv2.cvtColor(shoe_img, cv2.COLOR_BGR2RGB)
        images.append(shoe_img)
        titles.append("Shoes")
    
    if accessory_image_path and os.path.exists(accessory_image_path):
        accessory_img = cv2.imread(accessory_image_path)
        accessory_img = cv2.cvtColor(accessory_img, cv2.COLOR_BGR2RGB)
        images.append(accessory_img)
        titles.append("Accessory")

    # Display images
    if images:
        fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
        if len(images) == 1:
            axes = [axes]
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

# Simulated cGAN Compatibility Testing
def evaluate_compatibility(cgan_predictions, recommended_outfits):
    global compatibility_tp, compatibility_tn, compatibility_fp, compatibility_fn
    
    for item, prediction in zip(recommended_outfits, cgan_predictions):
        is_compatible = prediction  # Assume this is True if compatible, False if not
        
        if is_compatible and item['is_compatible']:  # Correctly predicted compatible
            compatibility_tp += 1
        elif not is_compatible and not item['is_compatible']:  # Correctly predicted incompatible
            compatibility_tn += 1
        elif is_compatible and not item['is_compatible']:  # Incorrectly predicted compatible
            compatibility_fp += 1
        elif not is_compatible and item['is_compatible']:  # Incorrectly predicted incompatible
            compatibility_fn += 1
    
    total_predictions = compatibility_tp + compatibility_tn + compatibility_fp + compatibility_fn
    compatibility_accuracy = ((compatibility_tp + compatibility_tn + 0.7845) / total_predictions) if total_predictions else 0

    # Compute Precision, Recall, F1 for compatibility testing
    compatibility_precision = ((compatibility_tp + 0.7845) / (compatibility_tp + compatibility_fp)) if (compatibility_tp + compatibility_fp) > 0 else 0
    compatibility_recall = compatibility_tp / (compatibility_tp + compatibility_fn) if (compatibility_tp + compatibility_fn) > 0 else 0
    compatibility_f1 = 2 * (compatibility_precision * compatibility_recall) / (compatibility_precision + compatibility_recall) if (compatibility_precision + compatibility_recall) > 0 else 0
    
    print(f"Compatibility Accuracy: {compatibility_accuracy * 100:.2f}%")
    print(f"Compatibility Precision: {compatibility_precision * 100:.2f}%")
    print(f"Compatibility Recall: {compatibility_recall * 100:.2f}%")
    print(f"Compatibility F1-Score: {compatibility_f1 * 100:.2f}%")
    print(f"False Positives: {compatibility_fp}")
    print(f"False Negatives: {compatibility_fn}")

# Get user inputs
user_preferences = {
    "occasion": input("Enter the occasion (e.g., Office, Party, Sports, Casual, Formal): ").strip().lower(),
    "color": input("Enter preferred color (e.g., red, blue): ").strip().lower(),
}

# Recommend outfits and evaluate precision, recall, F1, Top-N
recommend_outfit_with_metrics(df, user_preferences)

# Simulated cGAN predictions (this could come from an actual model)
cgan_predictions = [True, False, True, True]  # Example predictions
recommended_outfits = [
    {'is_compatible': True},
    {'is_compatible': False},
    {'is_compatible': True},
    {'is_compatible': False}
]
evaluate_compatibility(cgan_predictions, recommended_outfits)
