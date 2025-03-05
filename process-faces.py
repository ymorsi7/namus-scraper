import os, pickle, face_recognition
from glob import glob
from functools import reduce
from PIL import Image
import argparse
import re

PROCESS_FEEDBACK_INTERVAL = 50

FILE_LOCATIONS = "./output/{type}/files/*/"
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]

FACES_DATA = "./output/{type}/FaceEncodings.dat"
FACES_OUTPUT = "./output/{type}/faces/{case_id}-{category}-face{index}.{extension}"
CASE_TYPES = {
    "MissingPersons": {
        "excluded": []
    },
    "UnidentifiedPersons": {
        "excluded": ["/Clothing/", "/Footwear/", "/OfficeLogo/"]
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description='Process faces from NamUs case images')
    parser.add_argument('--limit', type=int, help='Limit the number of images to process')
    return parser.parse_args()

def main():
    args = parse_args()
    
    for caseType in CASE_TYPES:
        print("Processing: {type}".format(type=caseType))

        print(" > Fetching image file paths")
        paths = getImageFilesForType(caseType)
        
        if args.limit:
            paths = paths[:args.limit]
            print(f" > Limited to {args.limit} images")
            
        print(" > Found %d files" % len(paths))

        # Create parent directory for faces
        base_faces_dir = os.path.dirname(
            FACES_OUTPUT.format(type=caseType, case_id="*", category="*", index="*", extension="*")
        )
        os.makedirs(base_faces_dir, exist_ok=True)
        
        dataFile = open(FACES_DATA.format(type=caseType), 'wb')

        print(" > Starting face extraction")
        processedFiles, facesFound = 0, 0
        for path in paths:
            try:
                # Extract case ID and category from the path
                # Expected format: .../output/{type}/files/{category}/{case}-{file}.{extension}
                path_parts = path.split('/')
                
                # Get file components
                category = "unknown"
                case_id = "unknown"
                
                # Look for category (which is folder name above the file)
                if len(path_parts) >= 2:
                    category = path_parts[-2]
                
                # Extract case ID from filename pattern
                filename = path_parts[-1]
                case_id_match = re.match(r'^([^-]+)-', filename)
                if case_id_match:
                    case_id = case_id_match.group(1)
                
                # Extract filename and extension
                pathParts = filename.split(".")
                fileName = pathParts[0] if len(pathParts) > 0 else "unknown"
                extension = pathParts[1] if len(pathParts) > 1 else "jpg"

                image = face_recognition.load_image_file(path)
                locations = face_recognition.face_locations(image)
                encodings = face_recognition.face_encodings(image, locations)

                if len(encodings):
                    pickle.dump({path: encodings}, dataFile)

                for index, location in enumerate(locations):
                    outputPath = FACES_OUTPUT.format(
                        type=caseType, 
                        case_id=case_id,
                        category=category,
                        index=index,
                        extension=extension
                    )
                    
                    # Skip if file already exists
                    if os.path.exists(outputPath):
                        continue
                        
                    top, right, bottom, left = location
                    face = Image.fromarray(image[top:bottom, left:right])
                    face.save(outputPath)
                    facesFound += 1

                processedFiles += 1
                if processedFiles % PROCESS_FEEDBACK_INTERVAL == 0:
                    print(
                        " > Processed {count} files with {faces} faces".format(
                            count=processedFiles, faces=facesFound
                        )
                    )
            except Exception as e:
                processedFiles += 1
                print(f" > Failed parsing path: {path}. Error: {str(e)}")

        dataFile.close()


def getImageFilesForType(caseType):
    imageExtensionPaths = [
        FILE_LOCATIONS.format(type=caseType) + "*." + extension
        for extension in IMAGE_EXTENSIONS
    ]

    filePaths = reduce(lambda output, path: output + glob(path), imageExtensionPaths, [])
    for excluded in CASE_TYPES[caseType]["excluded"]:
        filePaths = list(filter(lambda path: excluded not in path, filePaths))

    return list(filePaths)


main()
