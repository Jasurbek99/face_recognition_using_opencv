import cv2
import os

# Function to capture and save photos with user-defined name
def capture_photos(output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize camera
    cap = cv2.VideoCapture(0)

    # Prompt user to input name
    name = input("Enter your name: ")

    photo_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            continue

        # Display the camera feed
        cv2.imshow('Capture Image', frame)

        # Wait for spacebar to be pressed to capture image
        key = cv2.waitKey(1)
        if key == 32:  # Spacebar
            # Create name folder if it doesn't exist
            name_folder = os.path.join(output_folder, name)
            if not os.path.exists(name_folder):
                os.makedirs(name_folder)

            # Save image to name folder
            photo_path = os.path.join(name_folder, f'photo_{photo_count}.jpg')
            cv2.imwrite(photo_path, frame)
            print(f"Photo {photo_count + 1} captured and saved as {photo_path}")

            photo_count += 1

        # Break the loop if 'q' key is pressed
        elif key & 0xFF == ord('q'):
            break

    # Release the camera and close OpenCqDowV windows
    cap.release()
    cv2.destroyAllWindows()

# Define the output folder where photos will be saved
output_folder = "dataset"

# Capture and save photos with user-defined name
capture_photos(output_folder)