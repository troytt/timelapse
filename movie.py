import moviepy.editor as mpe
import os

_ignore_file = "onion.jpg"

def MakeMovie(folder_path, output_file):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            if filename != _ignore_file:
                images.append(folder_path + filename)
    print(images[0])
    # Create a video clip from the images.
    clip = mpe.ImageSequenceClip(images, fps=60)
    clip.write_videofile(output_file)

if __name__ == '__main__':
    folder_path = "/Users/taodao/Downloads/timelapse/20240112_233451/"
    output_file = "/Users/taodao/Downloads/timelapse/20240112_233451.mp4"
    MakeMovie(folder_path, output_file)