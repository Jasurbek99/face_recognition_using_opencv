import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from deepface import DeepFace
import cv2

# List of available models and distance metrics
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
metrics = ["cosine", "euclidean", "euclidean_l2"]

class CameraApp:
    def __init__(self, root, cameras):
        self.root = root
        self.cameras = cameras
        self.frames = {}
        self.video_capture = {cam_id: cv2.VideoCapture(cam_url) for cam_id, cam_url in cameras.items()}
        
        self.setup_gui()
        self.update_frames()
    def setup_gui(self):
        self.root.title("Multi-Camera Face Recognition")
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = screen_width // len(self.cameras)
        window_height = screen_height // 2

        self.frame_labels = {}
        a=0
        r=0
        c=0
        for cam_id in self.cameras.keys():
           
            self.frame_labels[cam_id] = ttk.Label(self.root)
            if a%2==0:
                r+=1
                c=0
            self.frame_labels[cam_id].grid(row=r, column=c, padx=10, pady=10)
            self.root.geometry(f"{window_width}x{window_height}")
            a+=1
            c+=1

    def update_frames(self):
        for cam_id, cap in self.video_capture.items():
            ret, frame = cap.read()
            finded=False
            if ret:
                people = DeepFace.find(img_path=frame, db_path="dataset/Jennet", model_name=models[2], distance_metric=metrics[0], enforce_detection=False)
                for person in people:
                    try:
                        x = person['source_x'][0]
                        y = person['source_y'][0]
                        w = person['source_w'][0]
                        h = person['source_h'][0]
                        name = person['identity'][0].split('/')[1]
                        print(name)
                        
                        if (name=="Jennet"):
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            # cv2.rectangle(frame,(0,0),(frame.size),(0,255,0),5)
                            cv2.putText(frame, name, (x, y), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
                            print("Yes")
                            finded=True
                    except:
                        print("Error processing frame from camera", cam_id)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image = Image.fromarray(frame)
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                window_width = screen_width // len(self.cameras)
                window_height = screen_height // 2

                image.thumbnail((window_width - 20, window_height - 20))
                photo = ImageTk.PhotoImage(image=image)
                self.frame_labels[cam_id].configure(image=photo)
                if finded:
                    self.frame_labels[cam_id].configure(borderwidth=4)
                self.frame_labels[cam_id].image = photo

        self.root.after(10, self.update_frames)

def main():
    root = tk.Tk()
    
    # Define cameras with their respective URLs
    cameras = {
        1: 0,
        2: 1
        # 3: "http://192.168.119.35:8080/video",
        # 4: 1
        # Add more cameras as needed
    }

    app = CameraApp(root, cameras)
    root.mainloop()

if __name__ == "__main__":
    main()
