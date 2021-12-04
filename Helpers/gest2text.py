#!/usr/bin/env python
# coding: utf-8

# # 1. Import and Install Dependencies

# In[1]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


# # 2. Keypoints using MP Holistic

# In[2]:


mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 


# In[3]:


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                 
    results = model.process(image)                
    image.flags.writeable = True                  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results


# In[5]:


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 


# # 3. Extract Keypoint Values

# In[19]:


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


# # 4. Setup Folders for Collection

# In[34]:


mylist = os.listdir("ISL Continents and Countries/")
# print(mylist)
ls = []
for x in mylist:
    ls.append(os.path.splitext(x)[0])


# In[35]:


ls


# In[24]:


DATA_PATH = os.path.join('MP_Data') 
actions = np.array(ls)
no_sequences = 30
sequence_length = 30


# In[36]:


DATA_PATH = os.path.join('Pre') 


# In[37]:


import shutil
shutil.rmtree(DATA_PATH)


# In[38]:


DATA_PATH = os.path.join('Pre') 
actions = np.array(ls)
no_sequences = 1
sequence_length = 30


# In[39]:


for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass


# # 5. Collect Keypoint Values for Training and Testing

# In[41]:


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        s="ISL Continents and Countries/"
        s = s+ action + '.mp4'
        cap = cv2.VideoCapture(s)
        for sequence in range(no_sequences):
            frame_num = 0
            while (True):
                ret, frame = cap.read()
                if ret == True:
                    image, results = mediapipe_detection(frame, holistic)
                    if not (results.left_hand_landmarks or results.right_hand_landmarks):
                        continue
                    draw_styled_landmarks(image, results)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break     
                    frame_num+=1
                else:
                    break
                    
    cap.release()
    cv2.destroyAllWindows()


# In[30]:


cap.release()
cv2.destroyAllWindows()


# In[42]:


DATA_PATH = os.path.join('MP_Data') 


# In[43]:


shutil.rmtree(DATA_PATH)


# In[44]:


DATA_PATH = os.path.join('MP_Data') 
actions = np.array(ls)
no_sequences = 30
sequence_length = 30


# In[45]:


for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass


# In[46]:


import shutil
for action in actions:
    path="Pre/"
    dpath="MP_Data/"
    path+=action+"/0/"
    dpath+=action+"/"
    prepointer=0
    newspointer=0
    newsframe=0
    preend=len(os.listdir(path))
    while(True):
        if(prepointer==preend):
            prepointer=0
        if(newsframe==sequence_length):
            newsframe=0
            newspointer+=1
        if(newspointer==no_sequences):
            break
        src = path+str(prepointer)+'.npy'
        
        dst =  dpath+str(newspointer)+'/'+str(newsframe)+'.npy'
        print(dst)
        t =shutil.copyfile(src, dst)
        prepointer+=1
        newsframe+=1
    


# # 6. Preprocess Data and Create Labels and Features

# In[47]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# In[48]:


label_map = {label:num for num, label in enumerate(actions)}


# In[49]:


label_map


# In[50]:


sequences, labels = [], []
lis = []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
            lis.append(res)
        sequences.append(window)
        labels.append(label_map[action])


# In[51]:


np.array(sequences).shape


# In[53]:


X = np.array(sequences)


# In[54]:


X.shape


# In[55]:


y = to_categorical(labels).astype(int)


# In[56]:


y


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


# In[58]:


y_test.shape


# # 7. Build and Train LSTM Neural Network

# In[59]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


# In[60]:


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


# In[61]:


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


# In[62]:


res = [.7, 0.2, 0.1]


# In[63]:


actions[np.argmax(res)]


# In[64]:


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# In[66]:


model.fit(X_train, y_train, epochs=500,batch_size=100, callbacks=[tb_callback])


# In[67]:


model.summary()


# # 8. Make Predictions

# In[68]:


res = model.predict(X_test)


# In[69]:


actions[np.argmax(res[4])]


# In[70]:


actions[np.argmax(y_test[4])]


# # 9. Save Weights

# In[71]:


model.save('midsem.h5')


# In[ ]:


del model


# In[13]:


from tensorflow import keras
model = keras.models.load_model('action.h5')


# In[12]:


model.load_weights('action.h5')


# # 10. Evaluation using Confusion Matrix and Accuracy

# In[72]:


from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


# In[73]:


yhat = model.predict(X_test)


# In[74]:


ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()


# In[75]:


multilabel_confusion_matrix(ytrue, yhat)


# In[77]:


accuracy_score(ytrue, yhat)


# # 11. Test in Real Time

# In[85]:


colors = []
for i in actions:
    colors.append((245,117,16))
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    return output_frame


# In[91]:



import math
sequence = []
sentence = []
predictions = []
threshold = 0.3
frameRate = cap.get(5)
#ISL Continents and Countries/Mali.mp4
cap = cv2.VideoCapture('ISL Continents and Countries/Croatia.mp4')
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if(ret==True):
            image, results = mediapipe_detection(frame, holistic)
            if not (results.left_hand_landmarks or results.right_hand_landmarks):
                continue
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 

                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])
                if len(sentence) > 3: 
                    sentence = sentence[-3:]
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            i=0
            while(len(sequence)<30):
                sequence.append(sequence[i])
                i+=1
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 30: 
                    sentence = sentence[-3:]
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


# In[88]:


cap.release()
cv2.destroyAllWindows()


# In[ ]:




