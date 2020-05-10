import cv2
import os


# This code piece converts all images in a directory to GRAY scale image...
location_name = "VehicleDataRoad4Validation"

file_list = os.listdir(location_name)
image_files = [val[:-4] for val in file_list if val.__contains__(".png")]

for image in image_files:
    current_image = cv2.imread(location_name+"/" + image + ".png")
    gray_scale = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(location_name+"/" + image+"GRAY" + ".png", gray_scale)