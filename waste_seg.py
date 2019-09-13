import glob
import pandas as pd
from sklearn.utils import shuffle
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np

df=pd.read_csv("all_img.csv")
columns=["apples", "bananas", "cans", "cardboard", "oranges","plastic"]
datagen=ImageDataGenerator(rescale=1./255.)
test_datagen=ImageDataGenerator(rescale=1./255.)
train_generator=datagen.flow_from_dataframe(
dataframe=df[:933],
directory="location of dataset",
x_col="filename",
y_col="type",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
classes=["apples", "bananas", "cans", "cardboard", "oranges","plastic"],
target_size=(100,100))

valid_generator=test_datagen.flow_from_dataframe(
dataframe=df[933:985],
directory="location of dataset",
x_col="filename",
y_col="type",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
classes=["apples", "bananas", "cans", "cardboard", "oranges","plastic"],
target_size=(100,100))

test_generator=test_datagen.flow_from_dataframe(
dataframe=df[985:],
directory="location of dataset",
x_col="filename",
batch_size=1,
seed=42,
shuffle=False,
class_mode=None,
target_size=(100,100))


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(100,100,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='sigmoid'))


model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="binary_crossentropy",metrics=["accuracy"])


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
)
test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)
pred_bool = (pred >0.5)
final=[]
predictions = pred_bool.astype(int)
count=0
for i in predictions:
    if(i[0]==1):
        final.append(['apples'])
    elif(i[1]==1):
        final.append(['bananas'])
    elif(i[2]==1):
        final.append(['cans'])
    elif(i[3]==1):
        final.append(['cardboard'])
    elif(i[4]==1):
        final.append(['oranges'])
    elif(i[5]==1):
        final.append(['plastic'])  
    else:
        final.append(['none'])
final_result=[]        
for i in df[985:]['type']:
    final_result.append(i)
count=0
for i in range(len(final)):
    if(final[i]==final_result[i]):
        count=count+1

