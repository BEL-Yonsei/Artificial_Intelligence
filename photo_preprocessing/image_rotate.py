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

        # Apply the rotation to the image
        rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))

        # Save the rotated image to the disk
        ratio = str(int(angle))
        filename_parts = dest.split('.')
        destination = filename_parts[0] + '_' + angle + '.' + filename_parts[1]
        cv2.imwrite(destination, rotated_image)
    else:
        print("Failed to load the image.")
