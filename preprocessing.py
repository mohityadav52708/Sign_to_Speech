import numpy as np
import cv2
import os
from image_processing import func

data_path = "data"
train_path = os.path.join(data_path, "train")
test_path = os.path.join(data_path, "test")

if not os.path.exists(train_path):
    print("Error: 'train' folder not found in the 'data' directory.")
    exit()

if not os.path.exists(test_path):
    print("Error: 'test' folder not found in the 'data' directory.")
    exit()

a = ['label']

for i in range(64 * 64):
    a.append("pixel" + str(i))

label = 0
var = 0
c1 = 0
c2 = 0

# Process training data
print("Processing training data...")
for (dirpath, dirnames, filenames) in os.walk(train_path):
    for dirname in dirnames:
        print("Processing class:", dirname)
        class_path = os.path.join(train_path, dirname)
        for file in os.listdir(class_path):
            var += 1
            actual_path = os.path.join(class_path, file)
            img = cv2.imread(actual_path, 0)
            bw_image = func(actual_path)
            c1 += 1
            cv2.imwrite(os.path.join(train_path, dirname, file), bw_image)

# Process testing data
print("Processing testing data...")
for (dirpath, dirnames, filenames) in os.walk(test_path):
    for dirname in dirnames:
        print("Processing class:", dirname)
        class_path = os.path.join(test_path, dirname)
        for file in os.listdir(class_path):
            var += 1
            actual_path = os.path.join(class_path, file)
            img = cv2.imread(actual_path, 0)
            bw_image = func(actual_path)
            c2 += 1
            cv2.imwrite(os.path.join(test_path, dirname, file), bw_image)

print("Total files processed:", var)
print("Number of files for training:", c1)
print("Number of files for testing:", c2)
