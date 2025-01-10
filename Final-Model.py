import pandas as pd
import webcolors
from scipy.spatial import distance
import os
import re
import cv2
import matplotlib.pyplot as plt
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

# Function to compute weighted similarity score based on color and category
def compute_similarity_score(user_rgb_tuple, item_rgb_tuple, occasion, item_occasion):
    # Color similarity (Euclidean distance)
    color_distance = distance.euclidean(user_rgb_tuple, item_rgb_tuple)

    # Occasion match (more weight to color match)
    occasion_match = 0 if occasion != item_occasion else 1

    # Weighted score: combine both color distance and occasion match
    similarity_score = (100 - color_distance) + (occasion_match * 50)  # More weight to color match
    return similarity_score

# Recommend an outfit based on user inputs
def recommend_outfit(df, user_inputs):
    # Filter based on occasion
    filtered_df = df[df['Occasion'].str.lower() == user_inputs['occasion'].lower()]
    
    if filtered_df.empty:
        print("No items match your preferences.")
        return None, None, None

    # Match color using closest RGB
    user_rgb = rgb_from_text(user_inputs['color'])
    if not user_rgb:
        return None, None, None

    user_rgb_tuple = (user_rgb.red, user_rgb.green, user_rgb.blue)

    # Calculate similarity scores for all items based on color and occasion
    filtered_df = filtered_df.copy()
    filtered_df['Similarity Score'] = filtered_df.apply(
        lambda row: compute_similarity_score(user_rgb_tuple, parse_rgb(row['Dominant Colour']), user_inputs['occasion'], row['Occasion']),
        axis=1
    )

    # Sort by similarity score (higher score is better)
    filtered_df = filtered_df.sort_values(by='Similarity Score', ascending=False)

    # Separate categories
    tops = filtered_df[filtered_df['Category'].str.lower() == 'tops']
    bottoms = filtered_df[filtered_df['Category'].str.lower() == 'bottoms']
    outerwear = filtered_df[filtered_df['Category'].str.lower() == 'outerwear']
    shoes = filtered_df[filtered_df['Category'].str.lower() == 'shoes']
    accessories = filtered_df[filtered_df['Category'].str.lower() == 'accessory']

    if tops.empty and outerwear.empty:
        print("No tops or outerwear available for this occasion.")
        return None, None, None

    if bottoms.empty:
        print("No bottoms available for this occasion.")
        return None, None, None

    # Use outerwear if no tops are available
    if tops.empty:
        tops = outerwear

    # Return top 3 outfits for each category
    tops = tops.head(3)
    bottoms = bottoms.head(3)
    shoes = shoes.head(3)
    accessories = accessories.head(3)

    return zip(tops.iterrows(), bottoms.iterrows()), shoes.iterrows(), accessories.iterrows()

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

# Get user inputs
user_preferences = {
    "occasion": input("Enter the occasion (e.g., Office, Party, Sports, Casual, Formal): ").strip().lower(),
    "color": input("Enter preferred color (e.g., red, blue): ").strip().lower(),
}

# Recommend outfits based on user inputs
outfit_iterator, shoes_iterator, accessories_iterator = recommend_outfit(df, user_preferences)

if outfit_iterator:
    for (top_index, top), (bottom_index, bottom) in outfit_iterator:
        shoe = next(shoes_iterator, (None, None))[1] if shoes_iterator else None
        accessory = next(accessories_iterator, (None, None))[1] if accessories_iterator else None

        print("\nRecommended Outfit:")
        print("Top/Outerwear:", top[['Image ID', 'Dominant Colour', 'Clothing Apparel', 'Category']])
        print("Bottom:", bottom[['Image ID', 'Dominant Colour', 'Clothing Apparel', 'Category']])
        if shoe is not None:
            print("Shoes:", shoe[['Image ID', 'Dominant Colour', 'Clothing Apparel', 'Category']])
        if accessory is not None:
            print("Accessory:", accessory[['Image ID', 'Dominant Colour', 'Clothing Apparel', 'Category']])

        # Display the images side by side
        top_image_path = f"C:\\Narasimha\\KLETU Related\\5th Semester Related\\CV\\Course Project\\Dataset\\polyvore Dataset\\small_image\\{top['Image ID']}.jpg"
        bottom_image_path = f"C:\\Narasimha\\KLETU Related\\5th Semester Related\\CV\\Course Project\\Dataset\\polyvore Dataset\\small_image\\{bottom['Image ID']}.jpg"
        shoe_image_path = f"C:\\Narasimha\\KLETU Related\\5th Semester Related\\CV\\Course Project\\Dataset\\polyvore Dataset\\small_image\\{shoe['Image ID']}.jpg" if shoe is not None else None
        accessory_image_path = f"C:\\Narasimha\\KLETU Related\\5th Semester Related\\CV\\Course Project\\Dataset\\polyvore Dataset\\small_image\\{accessory['Image ID']}.jpg" if accessory is not None else None

        display_outfit(top_image_path, bottom_image_path, shoe_image_path, accessory_image_path)

        # Check user satisfaction
        satisfied = input("Are you satisfied with this recommendation? (yes/no): ").strip().lower()
        if satisfied == 'yes':
            print("Great! Enjoy your outfit.")
            break
        else:
            print("Looking for the next outfit...")
else:
    print("No recommendations found.")
