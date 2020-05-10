import os
import cv2


# This code piece resizes all images in a folder to 100x100 size...
location_name = "VehicleDataRoad4Validation"

file_list = os.listdir(location_name)
image_files = [val[:-4] for val in file_list if val.__contains__(".png") and
               not val.__contains__("GRAY")]# val.__contains__("GRAY")]  # val.__contains__("GRAY")]

for image in image_files:
    current_image = cv2.imread(location_name+"/" + image + ".png")
    resized_scale = cv2.resize(current_image, (100, 100))
    cv2.imwrite(location_name+"/" + image + "Resized" + ".png", resized_scale)
