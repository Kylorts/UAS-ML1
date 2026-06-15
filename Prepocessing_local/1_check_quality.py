import cv2
import os
import shutil


source_root = "C:\\Users\\User\\Documents\\Kuliah\\Semester 6\\ML 1\\UAS ML\\dataset\\raw_dataset"


target_root = "C:\\Users\\User\\Documents\\Kuliah\\Semester 6\\ML 1\\UAS ML\\dataset\\cleaned_dataset"

os.makedirs(target_root, exist_ok=True)


blur_threshold = 100.0
min_bright = 40
max_bright = 220

total_pass = 0
total_skip = 0

print("quality check dataset")


for class_name in os.listdir(source_root):

    class_path = os.path.join(source_root, class_name)

    if not os.path.isdir(class_path):
        continue

    target_class_path = os.path.join(target_root, class_name)

    os.makedirs(target_class_path, exist_ok=True)

    for file_name in os.listdir(class_path):

        if not file_name.lower().endswith(('.png','.jpg','.jpeg')):
            continue

        path = os.path.join(class_path, file_name)

        img = cv2.imread(path)

        if img is None:
            continue

        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()


        brightness = gray.mean()

        is_blur = laplacian_var < blur_threshold
        is_extreme_light = brightness < min_bright or brightness > max_bright

        if not is_blur and not is_extreme_light:

            shutil.copy(path, os.path.join(target_class_path, file_name))
            total_pass += 1

        else:
            total_skip += 1


print("quality check done")
print("image passed :", total_pass)
print("image skipped :", total_skip)
print("output folder :", target_root)