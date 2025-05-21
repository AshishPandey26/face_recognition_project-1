import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

# Load data from folders
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
        img = cv2.resize(img, (50, 50))
        faces.append(img.flatten())
        labels.append(person_name)

print(f"Loaded {len(faces)} faces for training.")

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces, labels)

# Load face detector
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # keep this file in same dir

# Start webcam
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_in_frame = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_in_frame:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (50, 50)).flatten().reshape(1, -1)
        predicted_name = knn.predict(face)[0]

        # Draw and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
