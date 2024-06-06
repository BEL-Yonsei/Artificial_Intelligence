import cv2
import keyboard
import numpy as np

def photo_capture(sour,dest):
    # Open the video file
    video_capture = cv2.VideoCapture(sour)

    # Check if the video file opened successfully
    if not video_capture.isOpened():
        print("Error: Couldn't open the video file at "+sour)
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # If frame is not read correctly, break the loop
        if not ret:
            break
        
        frame_number = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
        print("Frame number:", frame_number)
        cv2.imshow('image', frame)
        while True:
            cv2.waitKey(1)
            if keyboard.is_pressed('left'): # Press LEFT to go back
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 2)
                break
            if keyboard.is_pressed('right'): # Press RIGHT to go forward
                break
            if keyboard.is_pressed('space'): # Press SPACE to save file
                frame_number_str = str(int(frame_number))
                filename_parts = dest.split('.')
                destination = filename_parts[0] + '_' + frame_number_str + '.' + filename_parts[1]
                cv2.imwrite(destination,frame)
                print(f'Saved as {destination}')
                break

    # Release the video capture object
    video_capture.release()

