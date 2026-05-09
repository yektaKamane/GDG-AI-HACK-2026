import os

# 1. The hardcoded list of phrases (2 for each CIFAR-10 class)
phrases = [
    "The massive jet soared through the clouds leaving a white trail.",
    "A twin-engine aircraft prepared for landing on the runway.",
    "The sleek red sports car sped down the highway.",
    "I parked the family sedan in the driveway.",
    "A small robin gathered twigs to build its nest.",
    "The parrot squawked loudly from its cage.",
    "My fluffy tabby purred as she curled up on the sofa.",
    "The stray kitten chased a ball of yarn.",
    "A large buck with wide antlers stood quietly in the forest.",
    "We saw a fawn eating grass near the edge of the woods.",
    "The golden retriever caught the frisbee in the park.",
    "A police hound sniffed out the hidden package.",
    "The green amphibian leaped off the lily pad into the pond.",
    "We could hear the toads croaking after the rainstorm.",
    "The majestic stallion galloped across the open field.",
    "The jockey saddled up his mare for the big race.",
    "The massive cargo vessel docked at the busy port.",
    "A luxurious cruise liner sailed across the ocean.",
    "The delivery van backed up to unload its packages.",
    "An eighteen-wheeler hauled lumber down the interstate."
]

# 2. Create a folder to store the text files (so they don't make a mess!)
output_dir = "cifar_sentences"
os.makedirs(output_dir, exist_ok=True)

# 3. Loop through the list and create a separate .txt file for each one
# enumerate(..., start=1) makes our counting start at 1 instead of 0
for index, phrase in enumerate(phrases, start=1):
    
    # Create the filename (e.g., sentence1.txt, sentence2.txt)
    filename = f"sentence{index}.txt"
    
    # Combine the folder name and file name
    filepath = os.path.join(output_dir, filename)
    
    # Write the single phrase to the file
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(phrase)
        
print(f"Success! Created {len(phrases)} separate text files inside the '{output_dir}' folder.")