import face_recognition
import numpy as np
from PIL import Image, ImageDraw

known_face_encodings = np.load('face_encodings.npy', allow_pickle=True)
known_face_names = np.load('names.npy', allow_pickle=True)

unknown_image = face_recognition.load_image_file("kotla.jpg")

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

students_present = []
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = str(known_face_names[best_match_index])
        students_present.append(name)
