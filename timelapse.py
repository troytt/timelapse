import cv2
import argparse
import os
from datetime import datetime


#ffmpeg -f image2 -pattern_type glob -i "*.jpg" -c:v libx264 -pix_fmt yuv420p -r 30 -vf scale=3968:2976 output.mp4

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

now = datetime.now() # current date and time
date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
# Arguments
dir_path = '/Users/taodao/Downloads/timelapse/2023.03.31_16.27.27_balanced'
ext = 'jpg'
output = 'out_'+date_time+'.mp4'
fps = 60

images = []
for f in os.listdir(dir_path):
    if f.endswith(ext):
        images.append(f)
images.sort()

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
#cv2.imshow('video',frame)
height, width, channels = frame.shape
print('height:', height, ' width:', width)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, fps, (width, height))

count = 0
for image in images:
    count += 1
    if count % 100 == 0:
        print(count)
    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) # Write out frame to video

    #cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))