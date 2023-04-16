import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

# CNN Based Model

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

model.summary()

# Train

train_datagen = image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2, 
    zoom_range = 0.2,
    horizontal_flip = True,
)

test_dataset = image.ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    'drive/MyDrive/Dataset/Train',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary'
)

train_generator.class_indices

validation_generator = test_dataset.flow_from_directory(
    'drive/MyDrive/Dataset/Val',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary'
)

hist = model.fit(
    train_generator,
    steps_per_epoch = 6,
    epochs = 10,
    validation_data = validation_generator,
    validation_steps = 2
)

model.save('Detection_Covid_19.h5')

model = load_model("Detection_Covid_19.h5")

import os
from tensorflow.keras.utils import load_img, img_to_array

train_generator.class_indices

y_actual = []
y_test = []

for i in os.listdir("drive/MyDrive/Dataset/Val/Normal"):
  img = load_img("drive/MyDrive/Dataset/Val/Normal/"+i,target_size=(224,224))
  img = img_to_array(img)
  img = np.expand_dims(img,axis=0)
  p = model.predict(img)
  y_test.append(p[0,0])
  y_actual.append(1)

for i in os.listdir("drive/MyDrive/Dataset/Val/Covid"):
  img = load_img("drive/MyDrive/Dataset/Val/Covid/"+i,target_size=(224,224))
  img = img_to_array(img)
  img = np.expand_dims(img,axis=0)
  p = model.predict(img)
  y_test.append(p[0,0])
  y_actual.append(0)

y_actual = np.array(y_actual)
y_test = np.array(y_test)

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,plot_confusion_matrix

confusion_matrix(y_actual,y_test, labels=(1,0))
tp, fn, fp, tn = confusion_matrix(y_actual,y_test, labels=(1,0)).ravel()
#plot_confusion_matrix(final_model,X_test,y_test, labels=(1,0))

history = hist
print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

import numpy as np
# from google.colab.patches import cv2_imshow
import cv2
from keras.preprocessing import image
#xtest_image = image.load_img('drive/MyDrive/Dataset/Prediction/IM-0544-0001.jpeg', target_size = (224, 224))
xtest_image = load_img('drive/MyDrive/Dataset/Val/Covid/f6575117.jpg', target_size = (224, 224))
xtest_image = img_to_array(xtest_image)
xtest_image = np.expand_dims(xtest_image, axis = 0)
results = model.predict(xtest_image)
# training_set.class_indices
# imggg = cv2.imread('drive/MyDrive/Dataset/Prediction/IM-0544-0001.jpeg')
imggg = cv2.imread('drive/MyDrive/Dataset/Val/Covid/f6575117.jpg')
print("This Xray Image is of positive covid-19 patient")
imggg = np.array(imggg)
imggg = cv2.resize(imggg,(400,400))
plt.imshow(imggg)
# cv2_imshow(imggg)
# print(results)
if results[0][0] == 0:
    prediction = 'Positive For Covid-19'
else:
    prediction = 'Negative for Covid-19'
print("Prediction Of Our Model :",prediction)

import numpy as np
# from google.colab.patches import cv2_imshow
import cv2
from keras.preprocessing import image
xtest_image = load_img('drive/MyDrive/Dataset/Prediction/IM-0010-0001.jpeg', target_size = (224, 224))
xtest_image = img_to_array(xtest_image)
xtest_image = np.expand_dims(xtest_image, axis = 0)
results = model.predict(xtest_image)
# training_set.class_indices
imggg = cv2.imread('drive/MyDrive/Dataset/Prediction/IM-0010-0001.jpeg')
print("This Xray Image is of positive covid-19 patient")
imggg = np.array(imggg)
imggg = cv2.resize(imggg,(400,400))
plt.imshow(imggg)
# cv2_imshow(imggg)
# print(results)
if results[0][0] == 0:
    prediction = 'Positive For Covid-19'
else:
    prediction = 'Negative for Covid-19'
print("Prediction Of Our Model :",prediction)

import numpy as np
# from google.colab.patches import cv2_imshow
import cv2
from keras.preprocessing import image
xtest_image = load_img('drive/MyDrive/Dataset/Val/Normal/NORMAL2-IM-1117-0001.jpeg', target_size = (224, 224))
xtest_image = img_to_array(xtest_image)
xtest_image = np.expand_dims(xtest_image, axis = 0)
results = model.predict(xtest_image)
# training_set.class_indices
imggg = cv2.imread('drive/MyDrive/Dataset/Val/Normal/NORMAL2-IM-1117-0001.jpeg')
print("This Xray Image is of negative covid-19 patient")
imggg = np.array(imggg)
imggg = cv2.resize(imggg,(400,400))
plt.imshow(imggg)
# cv2_imshow(imggg)
# print(results)
if results[0][0] == 0:
    prediction = 'Positive For Covid-19'
else:
    prediction = 'Negative for Covid-19'
print("Prediction Of Our Model :",prediction)
