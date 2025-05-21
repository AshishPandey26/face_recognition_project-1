import os
import cv2

data_path = 'data/'
faces = []
labels = []

for person_name in os.listdir(data_path):
    person_folder = os.path.join(data_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        try:
            img = cv2.resize(img, (50, 50))
            faces.append(img.flatten())
            labels.append(person_name)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
