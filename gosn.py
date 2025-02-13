import cv2
import os
import face_recognition
import numpy as np
import time

# Step 1: Creating database of pictures
def create_database(name, save_dir="face_database"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cap = cv2.VideoCapture(0)
    count = 0

    while count < 20:  # Captures 20 images for better accuracy
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Retrying...")
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            file_name = os.path.join(save_dir, f"{name}_{count}.jpg")
            cv2.imwrite(file_name, face_image)
            print(f"Saved {file_name}")
            count += 1
            if count >= 20:
                break

        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Face database created successfully!")

# Step 2: Detect and recognize faces 
def detect_and_recognize(save_dir="C:/Users/kumbh/OneDrive/Desktop/Smart_Home_Automation/face_database"):
    known_encodings = []
    known_names = []

    # Load saved images and encode them
    for file in os.listdir(save_dir):
        if file.endswith(".jpg"):
            image_path = os.path.join(save_dir, file)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(file.split("_")[0])

    cap = cv2.VideoCapture(0)  # Open the video capture device
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        return

    start_time = time.time()
    detected_names = set()  # To store names detected during 30 seconds

    while time.time() - start_time < 30:  # Loop for 30 seconds
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            name = "Unknown"
            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

            # Add recognized names to the set (ignoring "Unknown")
            if name != "Unknown":
                detected_names.add(name)

            # Draw a rectangle around the face and label it
            cv2.rectangle(frame, (left, top), (right, bottom),(0, 165, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

        # Show the live video feed
        cv2.imshow("Face Recognition - Live", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Final decision after 30 seconds
    if detected_names:
        print(f"Access Granted. Detected guests: {', '.join(detected_names)}")
    else:
        print("Access Denied. No recognized faces detected (Unknown).")

if __name__ == "__main__":
    print("1. Create a new database")
    print("2. Detect and recognize faces")
    choice = input("Enter your choice (1/2): ")

    if choice == "1":
        person_name = input("Enter the name of the person: ")
        create_database(person_name)
    elif choice == "2":
        detect_and_recognize()
    else:
        print("Invalid choice!")



# from flask import Flask, request, jsonify
# import cv2
# import os
# import face_recognition
# import numpy as np

# app = Flask(__name__)

# # Load known faces
# face_db_path = "face_database"
# known_encodings = []
# known_names = []

# for file in os.listdir(face_db_path):
#     if file.endswith(".jpg"):
#         image_path = os.path.join(face_db_path, file)
#         image = face_recognition.load_image_file(image_path)
#         encodings = face_recognition.face_encodings(image)
#         if len(encodings) > 0:
#             known_encodings.append(encodings[0])
#             known_names.append(file.split("_")[0])

# @app.route('/detect', methods=['POST'])
# def detect_face():
#     cap = cv2.VideoCapture(0)
#     detected_name = "Unknown"

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_locations = face_recognition.face_locations(rgb_frame)
#         face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#         for face_encoding in face_encodings:
#             matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
#             face_distances = face_recognition.face_distance(known_encodings, face_encoding)

#             if matches:
#                 best_match_index = np.argmin(face_distances)
#                 if matches[best_match_index]:
#                     detected_name = known_names[best_match_index]
#                     cap.release()
#                     return jsonify({"name": detected_name, "status": "recognized"})

#         cap.release()
#         return jsonify({"name": "Unknown", "status": "unrecognized"})

# @app.route('/decision', methods=['POST'])
# def owner_decision():
#     data = request.json
#     if data['decision'] == "allow":
#         return jsonify({"access": "granted"})
#     else:
#         return jsonify({"access": "denied"})

