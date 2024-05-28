import cv2
import numpy as np
import time
import keyboard

def process_image(source,dest):
    
    # Load the image
    img = cv2.imread(source)
    # file not found
    if img is None:
        print(f'{source} not found')
        return

    # Create a clone of the image to draw on
    img_clone = img.copy()

    cv2.namedWindow('image')

    # Initial point
    ix, iy = -1, -1
    shift_x,shift_y=0,0

    # Initial switch
    sw=False
    windowOpened=False

    # Save cutted image
    imageSliced=None

    # Image cutting by ratio
    WIDTH=1
    HEIGHT=1

    def image_ratio(ix,iy,x,y):
        nonlocal WIDTH,HEIGHT
        new_x=x
        new_y=y
        current_x=x-ix
        current_y=y-iy
        unit=0
        # calculate the unit for the ratio
        if(current_y==0):
            # division by zero
            return new_x,new_y
        if(abs(current_x/current_y)>WIDTH/HEIGHT):
            # x is longer
            unit=int(current_y/HEIGHT)
        else:
            # y is longer
            unit=int(current_x/WIDTH)
        unit=abs(unit)

        # apply the unit to the new_x and new_y
        if(current_x>0):
            # new_x is bigger
            new_x=ix+unit*WIDTH
        else:
            # new_x is smaller
            new_x=ix-unit*WIDTH

        if(current_y>0):
            # new_y is bigger
            new_y=iy+unit*HEIGHT
        else:
            # new_y is smaller
            new_y=iy-unit*HEIGHT

        return new_x,new_y

    def image_position(ix,iy,x,y):
        if(ix>x):
            ms=ix
            ix=x
            x=ms
        if(iy>y):
            ms=iy
            iy=y
            y=ms
        return ix,iy,x,y

    # Image cutting  function
    def slice_image(img, start, end):
        # Check x and y to not twist it's coordinate
        for i in range(2):
            if(start[i]>end[i]):
                ms=start[i]
                start[i]=end[i]
                end[i]=ms

        # Return sliced image
        return img[start[1]:end[1], start[0]:end[0]]

    # Mouse callback function
    # click to start, click to end
    def draw_rectangle(event, x, y, flags, param):
        nonlocal ix, iy, img, img_clone,sw,windowOpened,imageSliced,shift_x,shift_y
        x+=shift_x
        y+=shift_y
        new_x,new_y=image_ratio(ix,iy,x,y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if sw :
                ix,iy,new_x,new_y=image_position(ix,iy,new_x,new_y)
                # (+1,+1,-1,-2) is for remove the rectangle border and suit to ratio
                imageSliced=slice_image(img_clone,[ix+1,iy+1],[new_x-1,new_y-1])
                cv2.imshow('example',imageSliced)
                windowOpened=True
                sw=False
            else :
                if(windowOpened):
                    cv2.destroyWindow('example')
                    windowOpened=False
                ix, iy = x, y
                sw=True
        elif event == cv2.EVENT_MOUSEMOVE:
            if ix != -1 and iy != -1:
                img_clone = img.copy()
                cv2.rectangle(img_clone, (ix, iy), (new_x, new_y), (255,255,0), 1)


    # Set parameter to use [sw, windowOpened, sliced image]
        # sw is switch of checking 'is it clicked?'
        # windowOpened is switch of checking 'is window opened?'
        # sliced image is image came from 
    param=[False,False,None]

    # Bind the function to window
    
    cv2.setMouseCallback('image', draw_rectangle,param)

    while(1):
        cv2.imshow('image', img_clone)
        cv2.waitKey(1)
        if keyboard.is_pressed('space'):  # Press SPACE to save file
            if windowOpened: # When the image is showing
                print('Saved')
                cv2.imwrite(dest,imageSliced)
        if keyboard.is_pressed('esc'):  # Press ESC to exit
            print('Exit!')
            break




    cv2.destroyAllWindows()
#process_image('Data/Original/0/10cm_+15deg.png','Data/Processed/test.png')