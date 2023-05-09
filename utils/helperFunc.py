import os
import cv2
import numpy as np


# frame width and height
frame_wid = 640
frame_hyt = 480

def imageLoader(path_arg):
    try:
        # Get a list of all items in the folder
        inter_path = path_arg
        real_path = ""
        for path in inter_path:
            real_path = real_path+path+" "
        # print(real_path)
        processed_folder_path = real_path.strip()
        items = os.listdir(processed_folder_path)
        print(f"[!] Found {len(items)} images [!]")

        images_path_list = []
        # Loop through each item in the folder
        for image in items:
            # Get the path of the item
            item_path = os.path.join(processed_folder_path, image)
            images_path_list.append(item_path)
        return (images_path_list, items)
    
    except Exception as err:
        print("[!] CHECK DATA FOLDER PATH ARGUMENT [!]") 

    # Return a tuple containing the list of image paths and the list of image names

def imagePreprosessing(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (frame_wid, frame_hyt))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray,(3,3))

    img_log = (np.log(blur+1)/(np.log(1+np.max(blur))))*255
    # Specify the data type
    img_log = np.array(img_log,dtype=np.uint8)

    bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)

    edges = cv2.Canny(bilateral,100,200)

    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return closing

def featureDetection(processsed_image):
    img = processsed_image
    # Create feature detecting method
    # sift = cv2.xfeatures2d.SIFT_create()
    # surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create(nfeatures=1500)

    # Make featured Image
    keypoints, descriptors = orb.detectAndCompute(img, None)
    featuredImg = cv2.drawKeypoints(img, keypoints, None)

    return featuredImg



