from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import face_recognition
import os

app = Flask(__name__)

# Load registered faces and their encodings
def load_registered_faces():
    registered_faces = {}
    face_database_dir = "face_database"
    if os.path.exists(face_database_dir):
        for person_name in os.listdir(face_database_dir):
            person_dir = os.path.join(face_database_dir, person_name)
            if os.path.isdir(person_dir):
                registered_faces[person_name] = []
                for image_name in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_name)
                    image = face_recognition.load_image_file(image_path)
                    face_encoding = face_recognition.face_encodings(image)
                    if face_encoding:
                        registered_faces[person_name].append(face_encoding[0])
    return registered_faces

registered_faces = load_registered_faces()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_face():
    data = request.json
    img_bytes = base64.b64decode(data['image'].split(",")[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not face_encodings:
        return jsonify({"name": "Unknown", "status": "no faces detected"})

    # Compare detected face with registered faces
    detected_name = "Unknown"
    for name, encodings in registered_faces.items():
        matches = face_recognition.compare_faces(encodings, face_encodings[0], tolerance=0.5)
        if True in matches:
            detected_name = name
            break

    return jsonify({"name": detected_name, "status": "face found", "count": len(face_encodings)})

@app.route('/register_face_batch', methods=['POST'])
def register_face_batch():
    data = request.json
    images = data['images']
    name = images[0]['name'].split('_')[0]

    # Create directory for person if not exists
    person_dir = os.path.join("face_database", name)
    os.makedirs(person_dir, exist_ok=True)

    saved_images = 0
    for i, img_data in enumerate(images):
        image_data = img_data['image']
        
        # Decode the image
        img_bytes = base64.b64decode(image_data.split(",")[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Detect face in the image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if face_locations:
            # Save the image
            file_path = os.path.join(person_dir, f"{name}_{saved_images}.jpg")
            cv2.imwrite(file_path, frame)
            saved_images += 1

    # Reload registered faces after new registration
    global registered_faces
    registered_faces = load_registered_faces()

    return jsonify({
        "message": "Images registered successfully", 
        "saved_count": saved_images
    })

if __name__ == "__main__":
    os.makedirs("face_database", exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)