import os
from pillow_heif import register_heif_opener
from PIL import Image

def convert_heic_to_jpeg(input_dir, output_dir):
    # Register HEIF opener with Pillow
    register_heif_opener()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all HEIC files in the input directory
    heic_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.heic')]
    
    print(f"Found {len(heic_files)} HEIC files to convert")
    
    # Convert each HEIC file to JPEG
    for heic_file in heic_files:
        input_path = os.path.join(input_dir, heic_file)
        output_filename = os.path.splitext(heic_file)[0] + '.jpg'
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            # Open and convert the image
            image = Image.open(input_path)
            image.convert('RGB').save(output_path, 'JPEG')
            print(f"Converted {heic_file} to {output_filename}")
        except Exception as e:
            print(f"Error converting {heic_file}: {str(e)}")

if __name__ == "__main__":
    input_directory = "safeway_data"
    output_directory = "safeway_data_jpeg"
    
    convert_heic_to_jpeg(input_directory, output_directory)
    print("Conversion complete!") 