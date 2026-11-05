import os

# Must be set BEFORE importing cv2: stops the slow MSMF hardware-transform
# startup that can make the camera take a minute+ to open on Windows.
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

t0 = time.time()
import cv2
import face_recognition
import numpy as np

print(f"imports: {time.time() - t0:.1f}s")

# name -> image file. Add more people here.
PEOPLE = {
    "Shailen": "shailen.jpg",
    # "Joe Biden": "biden.jpg",
}

known_face_encodings = []
known_face_names = []

t0 = time.time()
for name, path in PEOPLE.items():
    try:
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            print(f"No face found in {path}, skipping {name}.")
            continue
        known_face_encodings.append(encodings[0])
        known_face_names.append(name)
    except Exception as e:
        print(f"Could not load {path}: {e}")
print(f"encode known faces: {time.time() - t0:.1f}s")

if not known_face_encodings:
    raise SystemExit("No known faces loaded. Check your image files.")

t0 = time.time()
# CAP_DSHOW opens the camera far faster than the default backend on Windows.
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print(f"open camera: {time.time() - t0:.1f}s")

SCALE = 0.25  # detect on a quarter-size frame for speed
frame_count = 0
face_locations = []
face_names = []

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Could not read from camera.")
        break

    # Only run detection on every other frame.
    if frame_count % 2 == 0:
        small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)  # contiguous, dlib-safe

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            name = "Unknown"
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best = int(np.argmin(distances))
            if matches[best]:
                name = known_face_names[best]
            face_names.append(name)
    frame_count += 1

    inv = int(1 / SCALE)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top, right, bottom, left = top * inv, right * inv, bottom * inv, left * inv
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        if name == "Shailen":
            print("Recognized Shailen!")
            video_capture.release()
            cv2.destroyAllWindows()
            # You can add any action you want to perform when Shailen is recognized here.
        
    cv2.imshow("Video", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()