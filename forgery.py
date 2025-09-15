import os
os.getcwd()


# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
np.random.seed(2)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

from PIL import Image, ImageChops, ImageEnhance
import os
import itertools

def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    print(extrema)
    max_diff = max([ex[1] for ex in extrema])
    
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image

real_image_path = '/content/drive/MyDrive/Casia2/Au/Au_ani_30489.jpg'
Image.open(real_image_path)

convert_to_ela_image(real_image_path, 90)

fake_image_path = '/content/drive/MyDrive/Casia2/Tp/Tp_D_CRD_S_N_ani00075_art00012_00195.tif'
Image.open(fake_image_path)

convert_to_ela_image(fake_image_path, 90)

image_size = (128, 128)

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten()

X = [] # ELA converted images
Y = [] # 0 for fake, 1 for real

"""Dataset used: Casia

Total au images:7492

Total tp images:5124

Images are of different size and shape
"""

import random
path = '/content/drive/MyDrive/Casia2/Au'
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png'):
            full_path = os.path.join(dirname, filename)
            X.append(prepare_image(full_path))
            Y.append(1)
            if len(Y) % 500 == 0:
                print(f'Processing {len(Y)} images')

random.shuffle(X)
X = X[:2100]
Y = Y[:2100]
print(len(X), len(Y))

path = '/content/drive/MyDrive/Casia2/Tp'
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png'):
            full_path = os.path.join(dirname, filename)
            X.append(prepare_image(full_path))
            Y.append(0)
            if len(Y) % 500 == 0:
                print(f'Processing {len(Y)} images')

print(len(X), len(Y))

X = np.array(X)
Y = to_categorical(Y, 2)
X = X.reshape(-1, 128, 128, 3)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=5)
X = X.reshape(-1,1,1,1)
print(len(X_train), len(Y_train))
print(len(X_val), len(Y_val))

def build_model():
    model = Sequential()

    model.add(Conv2D(32, (5,5), padding = 'valid', activation = 'relu', input_shape = (128, 128, 3)))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3,3), padding = 'valid', activation = 'relu', input_shape = (128, 128, 3)))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'softmax'))
    
    return model

model = build_model()
model.summary()                #overview of the size of the different layers and the number of parameters the model has

epochs = 30
batch_size = 30

init_lr = 1e-4                                                  #lr decides how much gradient to be back propagated
optimizer = Adam(lr = init_lr, decay = init_lr/epochs)          #update network weights iterative based in training data

model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])      

#optimizer: optimize the input weights by comparing the prediction and the loss function.
#binary_crossentropy: Used as a loss function for binary classification model. It computes the cross-entropy loss between true labels and predicted labels.

early_stopping = EarlyStopping(monitor = 'val_acc',            #stop training once the model performance stops improving on the validation dataset
                              min_delta = 0,                   #consider an improvement that is a specific increment
                              patience = 2,
                              verbose = 0,                     #discover the training epoch on which training was stopped
                              mode = 'auto')

hist = model.fit(X_train,
                 Y_train,
                 batch_size = batch_size,
                 epochs = epochs,
                validation_data = (X_val, Y_val),
                callbacks = [early_stopping])

model.save('model_casia_run1.h5')

# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(hist.history['loss'], color='g', label="Training loss")
ax[0].plot(hist.history['val_loss'], color='y', label="Validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(hist.history['accuracy'], color='g', label="Training accuracy")
ax[1].plot(hist.history['val_accuracy'], color='y',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))  
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
  # xticks and yticks fucntion takes a list of objects as an argument and the objects denotes the position the data points at a specific positon 

    if normalize: 
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
#np.argmax() function returns indices of the max element of the array in a particular axis
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(2))

class_names = ['fake', 'real']

image_path = '/content/img2.png'
image = prepare_image(image_path)
image = image.reshape(-1, 128, 128, 3)
y_pred = model.predict(image)
y_pred_class = np.argmax(y_pred, axis = 1)[0]  #np.argmax() function returns indices of the max element of the array in a particular axis
print(f'The given image is: {class_names[y_pred_class]} image')
Image.open(image_path)