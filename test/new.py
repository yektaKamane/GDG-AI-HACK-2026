import pandas as pd
import numpy as np
from PIL import Image
import os

# Define the zip file and the folder where you want to save the images
csv_zip_path = 'train.csv.zip'
output_dir = 'cifar10_samples'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Loading {csv_zip_path}... This might take a minute.")
# Pandas can natively read compressed zip files
df = pd.read_csv(csv_zip_path)

# Assuming the column with the class name is called 'label'
# This grabs the first 5 rows for every unique label it finds
samples = df.groupby('label').head(5)

print("Extracting and saving images...")
saved_count = 0

for index, row in samples.iterrows():
    label = row['label']
    
    # Isolate just the pixel data by dropping the 'label' column
    # (If your CSV also has an 'id' column, add it here: row.drop(['label', 'id']))
    pixel_data = row.drop('label').values.astype(np.uint8)
    
    # CIFAR-10 data is usually structured as 1024 Red, 1024 Green, and 1024 Blue pixels.
    # We reshape it into 3 channels of 32x32, then transpose it to (32, 32, 3) for standard image formatting.
    try:
        image_array = pixel_data.reshape(3, 32, 32).transpose(1, 2, 0)
    except ValueError:
        # Fallback: If the pixels are stored as RGB, RGB, RGB instead of grouped by color channel
        image_array = pixel_data.reshape(32, 32, 3)
    
    # Convert the raw numbers into a standard image object
    img = Image.fromarray(image_array)
    
    # Create a unique filename (e.g., "frog_1.jpg", "frog_2.jpg")
    # We use saved_count % 5 so the numbering resets for each class (0 to 4)
    filename = f"{label}_{saved_count % 5}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    # Save it to the folder!
    img.save(filepath)
    print(f"Saved: {filepath}")
    saved_count += 1

print(f"\nDone! Successfully extracted {saved_count} JPGs into the '{output_dir}' folder.")
