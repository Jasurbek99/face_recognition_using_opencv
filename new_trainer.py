import cv2
import numpy as np
from PIL import Image
import os
# Path for face image database
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    labels = []
    label_id = -1
    label_map = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("jpg") or file.endswith("png") or file.endswith('.JPG'):
                imagePath= path = os.path.join(root, file)
                PIL_img = Image.open(imagePath).convert('L') # grayscale
                img_numpy = np.array(PIL_img,'uint8')
                label = os.path.basename(root).replace(" ", "-").lower()
               
                if (labels.count(label) == 0):
                     labels.append(label)
                     label_id+=1
                id_ = label_id
                faces = detector.detectMultiScale(img_numpy)
                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(id_)
                print(label,id_) 
    index_element_dict = {index: element for index, element in enumerate(labels)}
    print("Ashakdakyny using.py fayla label= diyen yere kopyalap goy")
    print(index_element_dict)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.save('trainer.yml') 
# Print the numer of faces trained and end program

print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))