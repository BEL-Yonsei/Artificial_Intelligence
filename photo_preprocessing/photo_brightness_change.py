from PIL import Image, ImageEnhance
from IPython.display import display
import numpy as np
import cv2

brighter=1.3
darker=0.7
def photo_brightness_change(img_path:str,save_path:str,original_save:bool,birghterStack:int,darkerStack:int):
    global brighter,darker
    # Open an image file
    with Image.open(img_path) as img:
        # Create a Brightness object from the image
        darker_enhancer=ImageEnhance.Brightness(img)
        brighter_enhancer=ImageEnhance.Brightness(img)
        # Save the original image
        if original_save:
            img.save(f'{save_path}_original.png')

        # Make the image brighter
        for i in range(birghterStack):
            img_enhanced = brighter_enhancer.enhance(brighter)
            # Save the enhanced image
            img_enhanced.save(f'{save_path}_brighter{i+1}.png')
            brighter_enhancer=ImageEnhance.Brightness(img_enhanced)


        # Make the image darker
        for i in range(darkerStack):
            img_enhanced = darker_enhancer.enhance(darker)
            # Save the enhanced image
            img_enhanced.save(f'{save_path}_darker{i+1}.png')
            darker_enhancer=ImageEnhance.Brightness(img_enhanced)
