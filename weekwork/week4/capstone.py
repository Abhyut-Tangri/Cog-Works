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
    cutoff = np.max(np.array(cosine_dists_same)) + 0.1
    return cutoff

model = FacenetModel()
dataase = FaceDatabase()
match_emotion = ""
dataase.load("database.pkl")
i = 0

#Camera.camera.save_camera_config(port = 1, exposure= 0.2)
#img_array = Camera.camera.take_picture() Takes a single image
cam = cv2.VideoCapture(1)
plt.ion()
fig, ax = plt.subplots()
ret, frame = cam.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
im = ax.imshow(frame)
#find_emotion_image_path = "Rian4.png"
#full_image = Image.open(find_emotion_image_path).convert('RGB') This just takes a photo from the path
emotions = []
min_cosine_dist = 1000
box1 = patches.Rectangle((0,0), height= -1*(0), width=(0) , linewidth=0, edgecolor='r', facecolor='none')
text = ax.text(0, 0, match_emotion, fontsize=0, color='red', ha='center', va='bottom')
number_of_faces = 0
actual_box = []
actual_text = []
for a in range(600):
    for k in range(len(actual_box)):
        actual_box[k].set_visible(False)
    actual_box = []
    for k in range(len(actual_text)):
        actual_text[k].set_visible(False)
    actual_text = []
    ret, frame = cam.read() # Read a frame from the camera
    if not ret:
        print("Error: Could not read frame.")
        break
    
    if cv2.waitKey(1) == ord('q'): # Wait for 'q' key press
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for i in range(1):
        boxes, probabilities, landmarks = model.detect(frame)
        if  boxes is not None:
            number_of_faces = np.array(boxes).shape[0]
            if number_of_faces != 1:
                for index , box in enumerate(boxes):
                    min_cosine_dist = 1000
                    descriptors = model.compute_descriptors(frame, [box]) #Descriptor Vector
                    for emotion in dataase.get_profiles():
                        cutoff = cutoffs(dataase, emotion)
                        cosine_dist = pairwise_cosine_distance(dataase.get_average_descriptor(emotion), descriptors)
                        if cutoff > cosine_dist:
                            if cosine_dist < min_cosine_dist:
                                min_cosine_dist = cosine_dist
                                match_emotion = emotion
                        if min_cosine_dist == 1000:
                            match_emotion = "Unkwown"
                    box2 = patches.Rectangle((box[0], box[3]), height= -1*(box[3] - box[1]), width=(box[2] - box[0]) , linewidth=1, edgecolor='r', facecolor='none')
                    #text1 = ax.text(box[0], box[3] + 50, match_emotion, fontsize=12, color='white', ha='center', va='bottom')
                    actual_box.append(box2)
                    actual_text.append(ax.text(box[0], box[3] + 50, match_emotion, fontsize=12, color='white', ha='center', va='bottom'))
                    ax.add_patch(actual_box[index])
                        
            else:
                descriptors = model.compute_descriptors(frame, boxes)
                for emotion in dataase.get_profiles():
                    cutoff = cutoffs(dataase, emotion)
                    cosine_dist = pairwise_cosine_distance(dataase.get_average_descriptor(emotion), descriptors)
                    if cutoff > cosine_dist:
                        if cosine_dist < min_cosine_dist:
                            min_cosine_dist = cosine_dist
                            match_emotion = emotion
                    if min_cosine_dist == 1000:
                            match_emotion = "Unkwown"
                box1 = patches.Rectangle((boxes[0][0], boxes[0][3]), height= -1*(boxes[0][3] - boxes[0][1]), width=(boxes[0][2] - boxes[0][0]) , linewidth=1, edgecolor='r', facecolor='none')
                text = ax.text(boxes[0][0], boxes[0][3] + 50, match_emotion, fontsize=12, color='white', ha='center', va='bottom')
                ax.add_patch(box1)
    im.set_data(frame)
    plt.pause(0.00001)
    plt.show()
    if number_of_faces == 1:
        box1.set_visible(False)
        text.set_visible(False)
    plt.draw()

plt.close()