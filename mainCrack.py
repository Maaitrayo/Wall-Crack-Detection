import cv2
import sys
import os

from utils.helperFunc import imageLoader, imagePreprosessing, featureDetection

if __name__ == '__main__':
    try:

        inter_path = sys.argv[1:]
        print(inter_path)
        real_path = ""
        for path in inter_path:
            real_path = real_path+path+" "

        folder_path = real_path.strip()
  

    except Exception as error:
        print("[!] Some error has occured [!]")

    try:
        image_path_list, image_name_list = imageLoader(folder_path)
        for path, name in zip(image_path_list, image_name_list):
            processed = imagePreprosessing(path)
            Feature_img = featureDetection(processsed_image= processed)
            cv2.imshow(f"{name}", Feature_img)
            cv2.waitKey(0)

    except Exception as error:
        print("[!] Some error occured during processing [!]")