import cv2

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load trained face recognition model
recognizer.read("trainer.yml")

# Dictionary to map label id to person names
labels = {0: "Dowletgeldi", 1:"Selim", 2: "Merdan", 3: "Mahri", 4: "Jennet", 5:"Tuwaktach", 6:  "Hajy", 7:"Gurbanberdi",8:"Madina"}

# Function to draw bounding box and label
def draw_label(img, text, x, y):
    cv2.rectangle(img, (x, y), (x + 100, y + 30), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, text, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Crop face region from the frame
        face_roi = gray[y:y + h, x:x + w]

        # Recognize face using LBPH recognizer
        label_id, confidence = recognizer.predict(face_roi)
        print(f"Recognized person: {label_id} {confidence}")
        if confidence > 10:
        # Get the corresponding label (person name) from the labels dictionary
            label_text = labels.get(label_id, "unknown")

            # Draw bounding box and label on the frame
            draw_label(frame, label_text, x, y)
        else:
            draw_label(frame, "unknown", x, y)

            # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
