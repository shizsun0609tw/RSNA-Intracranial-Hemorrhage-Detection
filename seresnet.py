import cv2
import numpy as np
import os
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import optimizers, losses
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.backend import int_shape
from tensorflow.keras.utils import to_categorical, plot_model

def se_block(block_input, num_filters, ratio=8):                             # Squeeze and excitation block
	pool1 = GlobalAveragePooling2D()(block_input)
	flat = Reshape((1, 1, num_filters))(pool1)
	dense1 = Dense(num_filters//ratio, activation='relu')(flat)
	dense2 = Dense(num_filters, activation='sigmoid')(dense1)
	scale = multiply([block_input, dense2])
	
	return scale

def resnet_block(block_input, num_filters):                                  # Single ResNet block
	if int_shape(block_input)[3] != num_filters:
		block_input = Conv2D(num_filters, kernel_size=(1, 1))(block_input)
	
	conv1 = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(block_input)
	norm1 = BatchNormalization()(conv1)
	relu1 = Activation('relu')(norm1)
	conv2 = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(relu1)
	norm2 = BatchNormalization()(conv2)
	
	se = se_block(norm2, num_filters=num_filters)
	
	sum = Add()([block_input, se])
	relu2 = Activation('relu')(sum)
	
	return relu2

def se_resnet14():
	input = Input(shape=(128, 128, 3))
	conv1 = Conv2D(64, kernel_size=(7, 7), activation='relu', padding='same', kernel_initializer='he_normal')(input)
	pool1 = MaxPooling2D((2, 2), strides=2)(conv1)
	
	block1 = resnet_block(pool1, 64)
	block2 = resnet_block(block1, 64)
	
	pool2 = MaxPooling2D((2, 2), strides=2)(block2)
	
	block3 = resnet_block(pool2, 128)
	block4 = resnet_block(block3, 128)
	
	pool3 = MaxPooling2D((3, 3), strides=2)(block4)
	
	block5 = resnet_block(pool3, 256)
	block6 = resnet_block(block5, 256)
	
	pool4 = MaxPooling2D((3, 3), strides=2)(block6)
	flat = Flatten()(pool4)
	
	output = Dense(6, activation='softmax')(flat)
	
	model = Model(inputs=input, outputs=output)
	return model



## load training data###

f = open('groundTruth.txt', 'r')
img_path = './image/'
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
        
    img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
    x_train.append(img)
    y_train.append(temp[1])
    print('class', temp[1], 'img', temp[0], 'loaded, img shape = ', img.shape)
    data = f.readline()


x_train_np = np.array(x_train)

y_train_np = np.array(y_train)
y_train_np = y_train_np.astype('int')
y_one_hot = np.zeros([y_train_np.size, np.amax(y_train_np)+1])
y_one_hot[np.arange(y_train_np.size), y_train_np.reshape(1, y_train_np.size)]  = 1

shuffler = np.random.permutation(len(x_train_np))
x_shuffled = x_train_np[shuffler]
y_shuffled = y_one_hot[shuffler]


## prepare generator

batch_size=64
train_datagen = ImageDataGenerator(featurewise_center=True,
                                    featurewise_std_normalization=True,
                                   zoom_range=0.5,
                                   rotation_range = 30,
                                   horizontal_flip=True,
                                   validation_split=0.2)

train_datagen.fit(x_shuffled)

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


## construct model
model_seresnet = se_resnet14()
model_seresnet.compile(loss='categorical_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

## train model
nb_epochs = 50
model_seresnet.fit_generator(
    train_generator,
    validation_data = validation_generator, 
    epochs = nb_epochs)


## load testing data

x_test = []
for filename in os.listdir('./test_img/'):
    img = cv2.imread(os.path.join('./test_img/',filename))
    if img is not None:
        img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
        x_test.append(img)
# np.array(x_test).shape


x_test_np = np.array(x_test)
test_datagen = ImageDataGenerator(featurewise_center=True,
                                    featurewise_std_normalization=True)

test_datagen.fit(x_test_np)

test_generator = test_datagen.flow(
    x = x_test_np) # set as testing data

y_predict = model_seresnet.predict(test_generator)

## output prediction to csv

fn = sorted(os.listdir('./test_img/'))

ans_count = [0,0,0,0,0,0]

count = 0
with open('predict_test.csv', 'w', newline='') as csvfile:
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