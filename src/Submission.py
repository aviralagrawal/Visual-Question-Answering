
# coding: utf-8

# In[1]:


import os
import pickle
import cv2


# In[2]:


jsonfile = 'Questions.json'
folder = 'images/'


# In[3]:


image_dict = {}        

skipped = []
i=0

for filename in os.listdir(folder):
    if(i%50 == 0):
        print("{} images processed".format(i))
    i+=1
    if filename in image_dict:
      # print ("already in dict - moving on")
        continue
    try:
      # load an image from file
        image = cv2.imread(os.path.join(folder, filename))
    except:
        print("Error reading file: {}!!!".format(filename))
        skipped.append(filename)
        continue
    if image is not None:
        resized_image = cv2.resize(image, (100, 100)) 
        image_dict[filename] = resized_image
    else:
        skipped.append(filename)

print("{} files skipped:".format(len(skipped)))
for f in skipped:
    print("    {}".format(f))
print("dict created")


print("Saving image_dict.pickle")
with open('image_dict.pickle', 'wb') as handle:
    pickle.dump(image_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('image_dict.pickle saved')


# In[42]:


for img in image_dict.keys():
    image_dict[img] = image_dict[img]/255


# In[43]:


imageNamesX  = []
questionsNLX = []
indexlist = []

import json
QAs = json.load(open(jsonfile, 'r'))['questions']

for QA in QAs:
    img_name = QA["Image"]+".png"
    ques = QA["Question"]
    ind = QA["Index"]  
    
    if img_name not in image_dict:
        print("Skipping {} - not found in dict".format(img_name))
        continue
    
    imageNamesX.append(img_name)
    questionsNLX.append(ques)
    indexlist.append(ind)


# In[44]:


import keras


# In[45]:


from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
with open('word_index.pickle','rb') as handle:
    word_index = pickle.load(handle)
    
print('Pickles loaded')

questionsX = tokenizer.texts_to_sequences(questionsNLX)
max_length_of_text = 200
questionsX = pad_sequences(questionsX, maxlen=max_length_of_text)


# In[46]:


# vector embeddings

embeddings_index = {}
    
EMBEDDING_DIM = 200

embedding_matrix = None

print(">> Embedding Matrix Pickle found...")
with open('embedding_matrix.pickle', 'rb') as handle:
    embedding_matrix = pickle.load(handle)
print(">>> loaded!")


# In[47]:


def generator(image_dict, img_names, questions, batch_size):
    
    q_ptr = 0
    while True:
        image_inp = []
        q_inp = []
#         batch_labels = []
        for i in range(batch_size):
            if q_ptr == len(questions):
                q_ptr = 0
            index = q_ptr
#             import random
#             index= random.randint(0, len(questions)-1)
            # print(imageNamesX[q_ptr].shape)
            # print(questionsX[q_ptr])
            image_inp.append(image_dict[img_names[index]])
            q_inp.append(questions[index])
#             batch_labels.append(labels[index])
            q_ptr+=1
            
        yield [np.array(image_inp), np.array(q_inp)]


# In[48]:


num_words = 81
embedding_dim = 200
dropout_rate=0.6
num_classes=26
seq_length = max_length_of_text


# In[ ]:


# from keras.models import Sequential, Model
# from keras.layers import Input, Dense, Activation, Dropout, LSTM, Flatten, Embedding, Multiply, Concatenate, Conv2D, BatchNormalization, Bidirectional, Conv1D, GlobalMaxPool1D, MaxPool1D
# from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
# import keras


# def vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
   
#     print("Creating image model...")
#     img_model = Sequential()
#     img_model.add(Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu'))
#     img_model.add(BatchNormalization())
#     img_model.add(Conv2D(48, kernel_size=(3, 3), strides=2, activation='relu'))
#     img_model.add(BatchNormalization())
#     img_model.add(Conv2D(48, kernel_size=(3, 3), strides=2, activation='relu'))
#     img_model.add(BatchNormalization())
#     img_model.add(Conv2D(64, kernel_size=(3, 3), strides=2, activation='relu'))
#     img_model.add(BatchNormalization())
#     img_model.add(keras.layers.Flatten())
   
#     image_input = Input(shape=(100, 100, 3))
#     encoded_image = img_model(image_input)
   

#     print("Creating text model...")
#     txt_model = Sequential()
#     txt_model.add(Embedding(num_words, embedding_dim,
#         weights=[embedding_matrix], input_length=seq_length, trainable=False))
#     txt_model.add(Conv1D(128, kernel_size=8, padding='same', activation='relu'))
#     txt_model.add(GlobalMaxPool1D())
# #     txt_model.add(Dense(128,activation='softmax'))

# #     txt_model.add(LSTM(units=128, return_sequences=False, input_shape=(seq_length, embedding_dim)))
# #     txt_model.add(Dropout(dropout_rate))
# #     txt_model.add(AttentionDecoder(512, embedding_dim))
# #     txt_model.add(LSTM(units=512, return_sequences=False))
# #     txt_model.add(Dropout(dropout_rate))
# #     txt_model.add(Dense(1024, activation='tanh'))
   
#     question_input = Input(shape=(EMBEDDING_DIM, ), dtype='int32')
#     embedded_question = txt_model(question_input)
   
#     print(txt_model.summary())
   
#     print("Merging final model...")
#     merged = keras.layers.concatenate([encoded_image, embedded_question])
#     d1  = Dense(512, activation='relu')(merged)
#     dp1 = Dropout(dropout_rate)(d1)
# #     d2  = Dense(1000, activation='tanh')(dp1)
# #     dp2 = Dropout(dropout_rate)(d2)
#     output  = Dense(num_classes, activation='softmax')(dp1)
   
#     vqa_model = Model(inputs=[image_input, question_input], outputs=output)
   
#     vqa_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   
#     return vqa_model


# In[49]:


from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout, LSTM, Flatten, Embedding, Multiply, Concatenate, Conv2D, BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import keras


def vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    
    print("Creating image model...")
    img_model = Sequential()
    img_model.add(Conv2D(24, kernel_size=(3, 3), strides=2, activation='relu'))
    img_model.add(BatchNormalization())
    img_model.add(Conv2D(48, kernel_size=(3, 3), strides=2, activation='relu'))
    img_model.add(BatchNormalization())
    img_model.add(Conv2D(48, kernel_size=(3, 3), strides=2, activation='relu'))
    img_model.add(BatchNormalization())
    img_model.add(Conv2D(64, kernel_size=(3, 3), strides=2, activation='relu'))
    img_model.add(BatchNormalization())
    img_model.add(keras.layers.Flatten())
    
    image_input = Input(shape=(100, 100, 3))
    encoded_image = img_model(image_input)
    
    print(img_model.summary())

    print("Creating text model...")
    txt_model = Sequential()
    txt_model.add(Embedding(num_words, embedding_dim, 
        weights=[embedding_matrix], input_length=seq_length, trainable=False))
    txt_model.add(LSTM(units=128, return_sequences=False, input_shape=(seq_length, embedding_dim)))
    txt_model.add(Dropout(dropout_rate))
#     txt_model.add(AttentionDecoder(512, embedding_dim))
#     txt_model.add(LSTM(units=512, return_sequences=False))
#     txt_model.add(Dropout(dropout_rate))
#     txt_model.add(Dense(1024, activation='tanh'))
    
    question_input = Input(shape=(EMBEDDING_DIM, ), dtype='int32')
    embedded_question = txt_model(question_input)
    
    print(txt_model.summary())
    
    print("Merging final model...")
    merged = keras.layers.concatenate([encoded_image, embedded_question])
    d1  = Dense(512, activation='relu')(merged)
    dp1 = Dropout(dropout_rate)(d1)
#     d2  = Dense(1000, activation='tanh')(dp1)
#     dp2 = Dropout(dropout_rate)(d2)
    output  = Dense(num_classes, activation='softmax')(dp1)
    
    vqa_model = Model(inputs=[image_input, question_input], outputs=output)
    
    
#     fc_model = Sequential()
#     # fc_model.add(Merge([vgg_model, lstm_model], mode='mul'))
#     fc_model.add(Concatenate([img_model, txt_model]))
#     fc_model.add(Dropout(dropout_rate))
#     fc_model.add(Dense(1000, activation='tanh'))
#     fc_model.add(Dropout(dropout_rate))
#     fc_model.add(Dense(num_classes, activation='softmax'))
    
    vqa_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return vqa_model


# In[50]:


# from keras.models import load_model,load_weights

modelname = 'checkPw.cp'

model = vqa_model(embedding_matrix, num_words, EMBEDDING_DIM, max_length_of_text, dropout_rate, num_classes)
model.load_weights(modelname)


# In[73]:


import numpy as np

pred = model.predict_generator(generator(image_dict, imageNamesX, questionsX, 1), steps = len(questionsX))


# In[74]:


pred.shape


# In[75]:


import tensorflow as tf

pred = np.argmax(pred, axis=1)


# In[77]:


pred = list(pred)


# In[92]:


from sklearn.preprocessing import LabelEncoder


# In[93]:


with open('label_encoder.pickle','rb') as handle:
    label_encoder = pickle.load(handle)


# In[94]:


label_encoder.classes_


# In[95]:


pred = list(label_encoder.inverse_transform(pred))


# In[96]:


import pandas as pd

ans = pd.DataFrame()
ans['Index']=indexlist
ans['pred']=pred


# In[98]:


ans.to_csv('2015A7PS0036G.csv')

