import cv2
import numpy as np
import os
from pathlib import Path

_mean_gray = 120

def adjust_brightness(input_image):
    """
    Parameters:
        - input_image：要处理的图片
    
    Returns:
        处理后的图片
    """
    scale_factor = _mean_gray / np.mean(input_image)
    input_image = (input_image * scale_factor).astype(np.uint8)
    return input_image

def folder_adjust_brightness(input_folder, output_folder, filename_prefix):
    """
    Parameters:
        - input_folder：要处理的图片的路径
        - output_folder：处理后保存的路径
        - filename_prefix：处理后图片的前缀
    
    Returns:
        无返回值，但会将处理后的图像保存在指定路径下。
    """
    # 计算所有灰度图像的平均值

    # 对每张灰度图像进行亮度缩放
    i = 0
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            i = i + 1
            if i % 100 == 0:
                print(i, filename)
            input_image = cv2.imread(os.path.join(input_folder, filename))
            output_image = adjust_brightness(input_image)
            # 将缩放后的灰度图像转换回RGB格式并保存处理后的照片
            output_path = os.path.join(output_folder, filename_prefix + filename)
            cv2.imwrite(output_path, output_image)


if __name__ == '__main__':
    folder_path = "/Users/taodao/Downloads/timelapse/2023.04.13_18.24.41"
    balanced_folder_path = folder_path + "_brightness/"
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    Path(balanced_folder_path).mkdir(parents=True, exist_ok=True)
    # 遍历指定文件夹下的所有图片文件，进行白平衡调整并保存
    folder_adjust_brightness(folder_path, balanced_folder_path, "")