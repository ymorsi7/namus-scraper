import os, re, glob, face_recognition
from PIL import Image
import argparse

# Folder locations
OUTPUT_BASE = "./output"
FACES_OUTPUT = "./output/faces/namus{namus_id}-face{index}.{extension}"

def parse_args():
    parser = argparse.ArgumentParser(description='Direct face extraction from NamUs images')
    parser.add_argument('--limit', type=int, help='Limit the number of images to process')
    parser.add_argument('--input', default=None, help='Specific input folder to process')
    return parser.parse_args()

def extract_namus_id(filepath):
    """Extract the NamUs ID from the file path or name"""
    # Try to find a number in the path that looks like a NamUs ID
    match = re.search(r'/(\d+)[/-]', filepath)
    if match:
        return match.group(1)
    
    # Try filename
    filename = os.path.basename(filepath)
    match = re.search(r'(\d+)', filename)
    if match:
        return match.group(1)
    
    return "unknown"

def main():
    args = parse_args()
    
    # Create output directory
    faces_dir = os.path.join(OUTPUT_BASE, "faces")
    os.makedirs(faces_dir, exist_ok=True)
    
    # Find all images in the output directory
    if args.input:
        search_path = args.input
    else:
        search_path = OUTPUT_BASE
        
    print(f"Searching for images in: {search_path}")
    image_paths = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_paths.extend(glob.glob(f"{search_path}/**/*.{ext}", recursive=True))
    
    print(f"Found {len(image_paths)} images")
    
    if args.limit:
        image_paths = image_paths[:args.limit]
        print(f"Limited to {args.limit} images")
    
    processed = 0
    faces_found = 0
    skipped = 0
    errors = 0
    
    for path in image_paths:
        try:
            namus_id = extract_namus_id(path)
            if namus_id == "unknown":
                print(f"Warning: Could not extract NamUs ID from {path}")
            
            print(f"Processing image: {path} (NamUs ID: {namus_id})")
            
            # Extract faces
            image = face_recognition.load_image_file(path)
            face_locations = face_recognition.face_locations(image)
            
            print(f"  Found {len(face_locations)} faces")
            
            # Save each face
            for i, face_location in enumerate(face_locations):
                top, right, bottom, left = face_location
                face_image = image[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                
                extension = os.path.splitext(path)[1][1:].lower()
                if extension not in ['jpg', 'jpeg', 'png']:
                    extension = 'jpg'
                
                output_path = FACES_OUTPUT.format(
                    namus_id=namus_id,
                    index=i,
                    extension=extension
                )
                
                # Skip if face already exists
                if os.path.exists(output_path):
                    print(f"  Skip face {i+1} (already exists): {output_path}")
                    skipped += 1
                    continue
                
                pil_image.save(output_path)
                print(f"  Saved face {i+1}: {output_path}")
                faces_found += 1
            
            processed += 1
            if processed % 10 == 0:
                print(f"Progress: {processed} images processed, {faces_found} faces extracted")
                
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            errors += 1
    
    print("\nSummary:")
    print(f"Images processed: {processed}")
    print(f"Faces extracted: {faces_found}")
    print(f"Faces skipped (duplicates): {skipped}")
    print(f"Errors: {errors}")

if __name__ == "__main__":
    main()
