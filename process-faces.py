import os, re, glob, face_recognition
from PIL import Image
import argparse
import requests
import time

# Folder locations
OUTPUT_BASE = "./output"
FACES_OUTPUT = "./output/faces/namus{namus_id}-face{index}.{extension}"

def parse_args():
    parser = argparse.ArgumentParser(description='Direct face extraction from NamUs images')
    parser.add_argument('--limit', type=int, help='Limit the number of images to process')
    parser.add_argument('--input', default=None, help='Specific input folder to process')
    parser.add_argument('--model', default='hog', choices=['hog', 'cnn'], 
                      help='Face detection model: hog (faster) or cnn (more accurate)')
    parser.add_argument('--force', action='store_true', help='Force reprocessing of all images')
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

def make_request_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response
        except Exception as e:
            print(f"Request failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2)  # Wait before retrying

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
    
    # CRITICAL FIX: Exclude the faces output directory to avoid reprocessing
    for ext in ['jpg', 'jpeg', 'png']:
        found_paths = glob.glob(f"{search_path}/**/*.{ext}", recursive=True)
        # Filter out any paths that are in the faces directory
        filtered_paths = [p for p in found_paths if '/faces/' not in p]
        image_paths.extend(filtered_paths)
    
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
            
            # Only log processing starts for larger batches to reduce output
            if processed % 10 == 0:
                print(f"Processing image: {path} (NamUs ID: {namus_id})")
            
            # Extract faces - use CNN model if specified (more accurate but slower)
            image = face_recognition.load_image_file(path)
            face_locations = face_recognition.face_locations(
                image, 
                model="cnn" if args.model == "cnn" else "hog"
            )
            
            # Only log face counts for non-zero or debugging
            if len(face_locations) > 0:
                print(f"  Found {len(face_locations)} faces")
            
            # Save each face
            for i, face_location in enumerate(face_locations):
                top, right, bottom, left = face_location
                
                # Add padding around the face (20% on each side)
                height = bottom - top
                width = right - left
                
                # Calculate new dimensions with padding (20%)
                padding_v = int(height * 0.2)
                padding_h = int(width * 0.2)
                
                # Ensure we don't go outside the image boundaries
                new_top = max(0, top - padding_v)
                new_bottom = min(image.shape[0], bottom + padding_v)
                new_left = max(0, left - padding_h)
                new_right = min(image.shape[1], right + padding_h)
                
                face_image = image[new_top:new_bottom, new_left:new_right]
                pil_image = Image.fromarray(face_image)
                
                extension = os.path.splitext(path)[1][1:].lower()
                if extension not in ['jpg', 'jpeg', 'png']:
                    extension = 'jpg'
                
                output_path = FACES_OUTPUT.format(
                    namus_id=namus_id,
                    index=i,
                    extension=extension
                )
                
                # Skip if face already exists and not forced
                if os.path.exists(output_path) and not args.force:
                    skipped += 1
                    continue
                
                pil_image.save(output_path)
                faces_found += 1
            
            processed += 1
            if processed % 100 == 0:
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


# python3 scrape-data.py
# python3 scrape-files.py
# python3 process-faces.py