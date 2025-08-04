import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from collections import Counter
import Camera.camera
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_models import FacenetModel
import cv2
import os
from fer import FER
import gradio as gr
import time


class Profile:

    """Stoes face descriptors for a emotiond individual."""

    def __init__(self,emotion):
        self.emotion=emotion 
        self.descriptors=dict()
    def add_descriptor(self,descriptor):
        """Adds a descriptor to the profile."""
        entry = {
            #np.asarray makes sure if the descriptor input is not a array, it will be through using that
            'emotion': self.emotion,
            'descriptor': np.asarray(descriptor)
        }
        self.descriptor.append(entry)
    def get_descriptor(self):
        return self.descriptors
    

class FaceDatabase:
    def __init__(self):
        self.profiles = {}
        self.cutoff = 0



    def add_image_descriptor(self, emotion, descriptor):
        if emotion not in self.profiles:
            self.profiles[emotion] = []
        self.profiles[emotion].append(descriptor)

    def save(self, path):

        with open(path, "wb") as f:
            pickle.dump(self.profiles, f)
        print(f"database saved to '{path}")
    
    def load(self, path):
        with open(path, "rb") as f:
            self.profiles = pickle.load(f)
        print(f"database loaded from '{path}")
    
    def get_average_descriptor(self,emotion):
        if emotion not in self.profiles:
            print("error: profile does not exist")
            return None
        
        if len(self.profiles[emotion]) == 0:
            print("error: profile has no descriptors")
            return None
        
        return np.mean(self.profiles[emotion], axis = 0)

    def get_profiles(self):
        return self.profiles


def pairwise_cosine_distance(M_descriptor, N_descriptor):
    """
    M_descriptor (np.ndarray): shape (M,D)
    N_descriptor (np.ndarray): shape (N,D)

    return : np.ndarray with shape (M, N)
    """
    #Normalize the vectors
    normed_M = M_descriptor / np.linalg.norm(M_descriptor)
    normed_N = N_descriptor / np.linalg.norm(N_descriptor)

    #Calculating dot product of normalized vectors
    cos_similarity = normed_M @ normed_N.T

    #Cosine Distance formula 
    cos_distance = 1 - cos_similarity

    return cos_distance

def cutoffs1(database, emotion):
    """
        Takes the database and the emotion to find the cutoffs

        database: FaceDatabase
        Gets the profiles to find the descriptors people

        emotion: String
        Uses the emotion to find the descriptors used in the cosine distances

        Returns:
        cutoff: float
        The number for the person in the database


        for keys in profile:
        descriptor_list.append(profile[keys])
        for key in profile:
            if keys != key:
                descriptor_list_other = profile[key]
                for i in range(len(descriptor_list_other)):
                    for j in range(i + 1, len(descriptor_list_other)):
                        cosine_dist = pairwise_cosine_distance(descriptor_list, descriptor_list_other)
                        cosine_dists.append(cosine_dist)
    """
    profile = database.get_profiles()
    cosine_dists = []
    cosine_dists_same = []
    descriptor_list = []
    descriptor_list_full = []
    for key in profile:
        print(profile[key])
        descriptor_list.append(profile[key])
        descriptor_list = descriptor_list[0]

        cosine_dist = pairwise_cosine_distance(descriptor_list, descriptor_list)
        cosine_dists_same.append(cosine_dist)

        descriptor_list = []
    
    for ind, key in enumerate(profile.keys()):
        if key == emotion:
            count_cosine_dists_same = Counter(np.array(cosine_dists_same[ind]).ravel())
        else:
            cosine_dists = (np.concatenate([np.array(cosine_dists_same[ind-1]), np.array(cosine_dists_same[ind-2])]))
            count_cosine_dists = Counter(cosine_dists.ravel())
    mean = np.array(cosine_dists_same).mean()
    std = np.array(cosine_dists_same).std()
    cutoff = mean - 0.64 * std
    return cutoff

def find_best_match(profiles, descriptor, threshold=0.92):
    new_descriptor = descriptor.flatten().reshape(1, -1)
    best_emotion = "Unknown"
    best_distance = float('inf')

    for profile in profiles:
        #find avg distance
        avg_descriptor = np.mean(np.vstack(profile.descriptors), axis=0).reshape(1,-1)
        distance = pairwise_cosine_distance(new_descriptor, avg_descriptor)[0][0]

        #check to see if it's the smallest distance
        if distance < best_distance and distance <= threshold:
            best_distance = distance
            best_emotion = profile.emotion

    if best_emotion == "Unknown":
        return best_emotion, None
    
    return best_emotion, best_distance


def display_labeled_faces(image_path, database, detector, encoder, threshold=0.92):

    full_image = Image.open(image_path).convert('RGB') ##I think it needs to be RGB
    draw_image = full_image.copy()
    draw = ImageDraw.Draw(draw_image)

    boxes, _ = detector.detect(full_image) 
   


    if boxes is None:
        print("No faces detected.")
        return


    # Transform for face images
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


  

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face_crop = full_image.crop((x1, y1, x2, y2))
        face_tensor = transform(face_crop).unsqueeze(0) 

        with torch.no_grad():
            descriptor = encoder(face_tensor)[0]  

        matched_emotion = "Unknown"
        best_score = -1

        for emotion, descriptors in database.get_profiles().items():
            for saved_desc in descriptors:
                similarity = torch.nn.functional.cosine_similarity(
                    descriptor.unsqueeze(0), saved_desc.unsqueeze(0)
                ).item()
                if similarity > threshold and similarity > best_score:
                    best_score = similarity
                    matched_emotion = emotion

        # Draw the face box and label
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), matched_emotion, fill="red")


    draw_image.show()
# this will download the pretrained weights (if they haven't already been fetched)
# which should take just a few seconds
def cutoffs(database, emotion):
    """
        Takes the database and the emotion to find the cutoffs

        database: FaceDatabase
        Gets the profiles to find the descriptors people

        emotion: String
        Uses the emotion to find the descriptors used in the cosine distances

        Returns:
        cutoff: float
        The number for the person in the database


        for keys in profile:
        descriptor_list.append(profile[keys])
        for key in profile:
            if keys != key:
                descriptor_list_other = profile[key]
                for i in range(len(descriptor_list_other)):
                    for j in range(i + 1, len(descriptor_list_other)):
                        cosine_dist = pairwise_cosine_distance(descriptor_list, descriptor_list_other)
                        cosine_dists.append(cosine_dist)
    """
    profile = database.get_profiles()
    cosine_dists = []
    cosine_dists_same = []
    descriptor_list = profile[emotion]
    for index, descriptor1 in enumerate(descriptor_list):
        for descriptor in descriptor_list:
            cosine_dist = pairwise_cosine_distance(descriptor1, descriptor)
            cosine_dists_same.append(cosine_dist)
    for key in (profile):
        if key != emotion:
            descriptor_list = (profile[key])
            descriptor_list1 = profile[emotion]
            for index, descriptor1 in enumerate(descriptor_list):
                for descriptor2 in descriptor_list1:
                    cosine_dist1 = pairwise_cosine_distance(descriptor1, descriptor2)
                    cosine_dists.append(cosine_dist1)
    cutoff = np.max(np.array(cosine_dists_same)) + 0.05
    return cutoff

model = FacenetModel()
dataase = FaceDatabase()
detector = FER()
"""
database_emotions = {}
emotions = {
    "anger": "images/train/angry",
    "disgust": "images/train/disgust",
    "fear" : "images/train/fear",
    "happy":"images/train/happy",
    "sad" : "images/train/sad",
    "suprise" : "images/train/surprise"
}
for emotion in emotions:
    count = 0
    emotion_path = emotions[emotion]
    for emote in os.listdir(emotion_path):
        if count < 150:
            if emote.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                file_path = os.path.join(emotion_path, emote)
                emotion_img = Image.open(file_path).convert('RGB')
                boxes, probabilities, landmarks = model.detect(emotion_img)
                if boxes is not None:
                    emotion_descriptor = model.compute_descriptors(emotion_img, boxes)
                    dataase.add_image_descriptor(emotion, emotion_descriptor)
        count += 1

dataase.save("database_emotions.pkl")
print("database saved!")
"""
def video(frame):
        match_emotion = ""
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, probabilities, landmarks = model.detect(frame_rgb)
        number_of_faces = 0

        if boxes is not None:
            number_of_faces = np.array(boxes).shape[0]

            if number_of_faces != 1:
                for box in boxes:
                    min_cosine_dist = 1000
                    descriptors = model.compute_descriptors(frame_rgb, [box])

                    for emotion in dataase.get_profiles():
                        cutoff = cutoffs(dataase, emotion)
                        cosine_dist = pairwise_cosine_distance(dataase.get_average_descriptor(emotion), descriptors)
                        if cutoff > cosine_dist and cosine_dist < min_cosine_dist:
                            min_cosine_dist = cosine_dist
                            match_emotion = emotion

                    if min_cosine_dist == 1000:
                        match_emotion = "Unknown"

                    # Draw box and label
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, match_emotion, (x1, y2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                match_emotion, score = detector.top_emotion(frame_rgb)
                box = boxes[0]
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, match_emotion, (x1, y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

#test accuracy of model
all_predictions= []
all_labels = []
emotions = {
    "anger": "images/validation/angry",
    "disgust": "images/validation/disgust",
    "fear" : "images/validation/fear",
    "happy":"images/validation/happy",
    "sad" : "images/validation/sad",
    "suprise" : "images/validation/surprise"
}
#in case the model returns an emotion not in the list, we will assign it to 'unknown'

emotion_list = list(emotions.keys()) + ["unknown"]
emotion_to_label = {emo: idx for idx, emo in enumerate(emotion_list)}
total=len(all_labels)
for emotion, folder in emotions.items():
    #line below may need to be tweaked idk the path structur
    for img_file in os.listdir(folder):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            img_path = os.path.join(folder, img_file)
            img = Image.open(img_path).convert('RGB')
            pred_emo, _ = detector.top_emotion(np.array(img))
            # If FER returns an emotion in your list, map it, else assign 'unknown'
            pred_label = emotion_to_label.get(pred_emo, emotion_to_label["unknown"])
            #append the prediction + the actual label
            all_predictions.append(pred_label)
            all_labels.append(emotion_to_label[emotion])
correct=0
# Calculate overall accuracy including 'unknown'
for guess, actual in zip(all_predictions, all_labels):
    if guess == actual:
        correct +=1

print(f"Test accuracy (including 'unknown' predictions): {accuracy * 100:.2f}%")

if total>0:
    accuracy = correct / total
else:
    accuracy = 0
print(f"Test accuracy : {accuracy * 100:.2f}%")



demo = gr.Interface(
    fn=video,
    inputs=gr.Image(streaming=True),
    outputs="image",
    live=True,
    title="Real-Time Emotion Recognition",
    description="Detects emotions live from webcam feed."
)

if __name__ == "__main__":
    demo.launch()


