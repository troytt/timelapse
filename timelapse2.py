import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/Users/taodao/audio-orchestrator-ffmpeg/bin/ffmpeg"

import movie
import brightness
import white_balance
import cv2
import numpy as np
import os
from pathlib import Path

_ignore_file = "onion.jpg"
_fix_img = False
if __name__ == '__main__':
    root_dir = "/Users/taodao/Downloads/timelapse/"
    names = ["20240224_202701"]

    for name in names:
        raw_folder_path = root_dir + name + "/"
        fixed_folder_path = root_dir + name + "_fixed/"
        output_file = root_dir + name +  ".mp4"
        Path(raw_folder_path).mkdir(parents=True, exist_ok=True)
        Path(fixed_folder_path).mkdir(parents=True, exist_ok=True)

        if not _fix_img:
          movie.MakeMovie(raw_folder_path, output_file)
          exit()

        i = 0
        for filename in os.listdir(raw_folder_path):
            if filename == _ignore_file:
                continue
            if filename.endswith(".jpg") or filename.endswith(".png"):
                i = i + 1
                if i % 100 == 0:
                    print(i, filename)
                input_image_path = os.path.join(raw_folder_path, filename)
                output_image_path = os.path.join(fixed_folder_path, filename)
                # Adjust image
                output_image = cv2.imread(input_image_path)
                # output_image = white_balance.adjust_white_balance(output_image)
                output_image = brightness.adjust_brightness(output_image)
                cv2.imwrite(output_image_path, output_image)
        movie.MakeMovie(fixed_folder_path, output_file)
