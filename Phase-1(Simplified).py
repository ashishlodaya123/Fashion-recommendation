import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import json

# Function to extract the dominant color from an image
def extract_dominant_color(image_path, k=3):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (600, 400))
    pixels = resized_image.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    most_common_color = np.round(kmeans.cluster_centers_).astype(int)[0]
    
    return most_common_color

# Paths for files
image_folder = "C:\\Narasimha\\KLETU Related\\5th Semester Related\\CV\\Course Project\\Dataset\\polyvore Dataset\\small_image"  # Folder with images
output_csv = 'C:\\Narasimha\\KLETU Related\\5th Semester Related\\CV\\Course Project\\Dataset\\polyvore Dataset\\Processed_Data.csv'  # Output CSV
json_file = "C:\\Narasimha\\KLETU Related\\5th Semester Related\\CV\\Course Project\\Dataset\\polyvore Dataset\\polyvore_item_metadata.json"  # JSON file with Image ID -> Category ID mapping
csv_file = "C:\\Narasimha\\KLETU Related\\5th Semester Related\\CV\\Course Project\\Dataset\\polyvore Dataset\\categories.csv"  # CSV file with Category ID -> Apparel mapping

# Step 1: Load JSON data (Image ID -> Category ID)
with open(json_file, 'r') as f:
    image_to_category = json.load(f)  # JSON format: {"3441": 56, "45673": 25, ...}

# Step 2: Load Category Mapping from CSV (Category ID -> Apparel and Category)
category_mapping = pd.read_csv(csv_file, header=None)  # Load the CSV file without headers
category_mapping.columns = ['Category ID', 'Clothing Apparel', 'Category']  # Manually assign column names

duplicates = category_mapping[category_mapping.duplicated(subset=['Category ID'], keep=False)]
print("Duplicates found:")
print(duplicates)

# Handle duplicate Category IDs by keeping the first occurrence
category_mapping = category_mapping.drop_duplicates(subset=['Category ID'])

# Convert to a dictionary
category_dict = category_mapping.set_index('Category ID')[['Clothing Apparel', 'Category']].to_dict(orient='index')


# List to hold image data
image_data = []

# Normalize category_dict keys to strings
category_dict = {str(key): value for key, value in category_dict.items()}

# Step 3: Process each image in the folder
for image_name in os.listdir(image_folder):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter image files
        image_path = os.path.join(image_folder, image_name)
        dominant_color = extract_dominant_color(image_path)
        image_id = os.path.splitext(image_name)[0]
        
        # Store dominant color
        color_str = f"RGB({dominant_color[0]}, {dominant_color[1]}, {dominant_color[2]})"
        
        # Get Category ID dictionary from JSON
        category_data = image_to_category.get(image_id)
        
        # Check if category_data is valid and extract category_id
        if isinstance(category_data, dict):
            category_id = category_data.get('category_id')
            category_id = str(category_id)  # Ensure category_id is a string
        else:
            category_id = None
        
        # Debugging: Print Image ID and extracted Category ID
        print(f"Processing Image ID: {image_id}, Extracted Category ID: {category_id}")
        
        # Get Clothing Apparel and Category from Category ID
        if category_id is not None:
            if category_id in category_dict:
                apparel_info = category_dict.get(category_id)
                clothing_apparel = apparel_info['Clothing Apparel']
                category = apparel_info['Category']
            else:
                print(f"Category ID {category_id} not found in category_dict keys")
                clothing_apparel = "Unknown"
                category = "Unknown"
        else:
            print(f"Category ID for Image ID {image_id} is missing or invalid")
            clothing_apparel = "Unknown"
            category = "Unknown"
        
        # Append data to the list
        image_data.append([image_id, color_str, clothing_apparel, category])

# Step 4: Create a DataFrame and save to CSV
df = pd.DataFrame(image_data, columns=['Image ID', 'Dominant Colour', 'Clothing Apparel', 'Category'])
df.to_csv(output_csv, index=False)

print(f"Dominant colors and categories saved to {output_csv}")
