from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import VGG16
net = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(128,128,3))
baseModel = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(128,128,3))
testModel = VGG16(weights='imagenet')
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.models import Model
pre_model = Model(testModel.input, testModel.get_layer('block5_conv3').output)
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(pre_model)
model.add(layers.Conv2D(256,3))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(512,3))

model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(layers.BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(256, activation="relu"))
model.add(layers.BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(6, activation="softmax", kernel_initializer='glorot_normal'))


for layer in model.layers[:-8]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate = 0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
import cv2
f = open('D:/desktop/med_hw2/groundTruth.txt', 'r')
img_path = 'D:/desktop/med_hw2/image/'
data = f.readline()
x_train = []
y_train = []
while data:
    temp = data.split(" ")
    temp[1] = temp[1].strip('\n')
    if temp[1] == '0':
        img = cv2.imread(img_path + 'epidural/' + temp[0])
    elif temp[1] == '1':
        img = cv2.imread(img_path + 'healthy/' + temp[0])
    elif temp[1] == '2':
        img = cv2.imread(img_path + 'intraparenchymal/' + temp[0])
    elif temp[1] == '3':
        img = cv2.imread(img_path + 'intraventricular/' + temp[0])
    elif temp[1] == '4':
        img = cv2.imread(img_path + 'subarachnoid/' + temp[0])
    elif temp[1] == '5':
        img = cv2.imread(img_path + 'subdural/' + temp[0])
        
    ###resize image here before appending，要把上面ResNet的input shape也改成這個shape###
    img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
    x_train.append(img)
    y_train.append(temp[1])
    #print('class', temp[1], 'img', temp[0], 'loaded, img shape = ', img.shape)
    data = f.readline()
import numpy as np
x_train_np = np.array(x_train)
y_train_np = np.array(y_train)
y_train_np = y_train_np.astype('int')
y_one_hot = np.zeros([y_train_np.size, np.amax(y_train_np)+1])
y_one_hot[np.arange(y_train_np.size), y_train_np.reshape(1, y_train_np.size)]  = 1
shuffler = np.random.permutation(len(x_train_np))
x_shuffled = x_train_np[shuffler]
y_shuffled = y_one_hot[shuffler]
from tensorflow.keras.preprocessing.image import ImageDataGenerator
batch_size=64
train_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2, rotation_range = 10)
train_generator = train_datagen.flow(
    x = x_shuffled,
    y = y_shuffled,
    batch_size=batch_size,
    subset='training') # set as training data

validation_generator = train_datagen.flow(
    x = x_shuffled,
    y = y_shuffled,
    batch_size=batch_size,
    subset='validation') # set as validation data

#training
nb_epochs = 5
model.fit_generator(
    train_generator,
    validation_data = validation_generator, 
    epochs = nb_epochs)

#testing
import os
x_test = []
for filename in os.listdir('D:/desktop/med_hw2/test_image/'):
    img = cv2.imread(os.path.join('D:/desktop/med_hw2/test_image/',filename))
    if img is not None:
        img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
        x_test.append(img)
x_test_np = np.array(x_test)
y_predict = model.predict(x_test_np)
import csv

fn = sorted(os.listdir('D:/desktop/med_hw2/test_image/'))

ans_count = [0,0,0,0,0,0]

count = 0
with open('predict.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in y_predict:
        ans_count[np.argmax(i)] = ans_count[np.argmax(i)]+1
        if np.argmax(i) == 0:
            writer.writerow([fn[count].split('.')[0], 'epidural'])
        elif np.argmax(i) == 1:
            writer.writerow([fn[count].split('.')[0], 'healthy'])
        elif np.argmax(i) == 2:
            writer.writerow([fn[count].split('.')[0], 'intraparenchymal'])
        elif np.argmax(i) == 3:
            writer.writerow([fn[count].split('.')[0], 'intraventricular'])
        elif np.argmax(i) == 4:
            writer.writerow([fn[count].split('.')[0], 'subarachnoid'])
        elif np.argmax(i) == 5:
            writer.writerow([fn[count].split('.')[0], 'subdural'])
        count = count+1
print(ans_count)