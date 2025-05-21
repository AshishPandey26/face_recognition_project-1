import cv2
import face_recognition
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# Load known faces and labels from folder
def load_known_faces(data_path='data'):
    known_encodings = []
    known_names = []

    for person_name in os.listdir(data_path):
        person_folder = os.path.join(data_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_file)
            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
    return known_encodings, known_names

print("Loading known faces...")
known_face_encodings, known_face_names = load_known_faces()
print(f"Loaded {len(known_face_names)} face samples.")

video = cv2.VideoCapture(0)

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Resize frame to speed up face detection (optional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        name = "Unknown"
        if matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Scale face locations back up because the frame we detected in was scaled down to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

        # Attendance recording info
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        attendance = [name, timestamp]
        attendance_file = f"Attendance/Attendance_{date}.csv"
        exist = os.path.isfile(attendance_file)

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('o'):
        speak("Attendance Taken")
        time.sleep(2)
        if not os.path.exists("Attendance"):
            os.makedirs("Attendance")
        if exist:
            with open(attendance_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(attendance)
        else:
            with open(attendance_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)

    elif key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
