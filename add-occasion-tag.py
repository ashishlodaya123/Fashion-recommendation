import pandas as pd

# Define the mapping logic for occasions
mapping = {
    "Party": ["dress", "gown", "heels", "clutch", "necklace", "earrings", "pumps", "bracelet", "all-body"],
    "Office": ["blazer", "trench coat", "pants", "skirt", "flats", "belt", "watch"],
    "Casual": ["jeans", "tank", "shorts", "sneakers", "sunglasses", "backpack"],
    "Formal": ["suit jacket", "blouse", "trousers", "heels", "flats", "scarf"],
    "Sports": ["socks", "sneakers", "headband", "flip-flops", "umbrella"],
}

# Function to map apparel and category to an occasion
def map_to_occasion(apparel, category):
    for occasion, items in mapping.items():
        if apparel.lower() in items or category.lower() in items:
            return occasion
    return "Unknown"

# Load your dataset (replace 'your_file.csv' with the actual file path)
df = pd.read_csv("C:\\Narasimha\\KLETU Related\\5th Semester Related\\CV\\Course Project\\Dataset\\polyvore Dataset\\Processed_Data-2.csv")

# Ensure columns are lowercase for consistency
df["Clothing Apparel"] = df["Clothing Apparel"].str.lower()
df["Category"] = df["Category"].str.lower()

# Apply the mapping function to create the Occasion column
df["Occasion"] = df.apply(lambda row: map_to_occasion(row["Clothing Apparel"], row["Category"]), axis=1)

# Save the updated dataset to a new file
df.to_csv("C:\\Narasimha\\KLETU Related\\5th Semester Related\\CV\\Course Project\\Dataset\\polyvore Dataset\\Data-with-occasion.csv", index=False)

print("Occasion column mapped and dataset saved.")
