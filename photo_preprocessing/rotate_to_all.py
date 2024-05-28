import os
import image_crop
import photo_brightness_change
# Directory paths
sourceDir = 'Data/Original'
targetDir = 'Data/Processed'

numbers=['0','1','2','3','4','5','6','7','8','9']
for num in numbers:
    # Get all files
    fileList = os.listdir(sourceDir+"/"+num)

    # Process each file
    for filename in fileList:
        # Construct the source and target file paths
        sourceFile = sourceDir+"/"+num+"/"+filename
        targetFile = targetDir+"/"+num+"/"+filename.split(".")[0]
        #image_crop.process_image(sourceFile, targetFile)
        photo_brightness_change.photo_brightness_change(sourceFile,targetFile,True,2,2)

    


