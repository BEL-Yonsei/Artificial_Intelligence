import cv2


def image_rotate(sour, dest, angle):
    # Read the image
    img = cv2.imread(sour)

    # Check if the image was loaded successfully
    if img is not None:
        # Get the image dimensions
        height, width = img.shape[:2]

        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

        # Apply the rotation to the image with black background
        rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height), borderValue=(0, 0, 0))

        # Save the rotated image to the disk
        ratio = str(int(angle))
        filename_parts = dest.split('.')
        destination = filename_parts[0] + '_' + str(angle) + '.' + filename_parts[1]
        cv2.imwrite(destination, rotated_image)
    else:
        print("Failed to load the image.")

# Example
# image_rotate('Data/Original/0_10cm_+30deg.png','Data/Processed/0_10cm_+30deg.png',45)
# will save 0_10cm_+30deg_45.png