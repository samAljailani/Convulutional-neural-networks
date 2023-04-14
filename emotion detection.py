from tensorflow.keras.layers              import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Dense, Activation, GlobalAveragePooling2D
from tensorflow.keras.optimizers          import Adam
from tensorflow.keras.models              import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks           import EarlyStopping
from sklearn.metrics import confusion_matrix


import os
import numpy as np
import matplotlib.pyplot as plt

################################################################
input_shape = (48,48,1)
n_output = 7

filters      = [32,64,128, 256]
conv_kernels = [(3,3), (3,3), (3,3), (3,3) , (3,3)]
neurons      = [512, 256, 128]
dropout      = [0.4, 0.4]
beta         = [0.95, 0.99]
weight_decay = 1e-06
batch_size = 64
n_epoch = 80

earlystop =  EarlyStopping( monitor="val_accuracy", patience=10, baseline= 0.02, restore_best_weights=True, verbose=1)
###################Model Architecture###########################
def build_model(filters, conv_kernels, dropout_rates, neurons, weight_decay, beta):
    model = Sequential()

    # First convolutional block
    model.add(Conv2D(filters[0], kernel_size = conv_kernels[0], padding="same", input_shape=(48,48,1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(filters[0], kernel_size = conv_kernels[0], padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rates[0]))

    # Second convolutional block
    model.add(Conv2D(filters[1], conv_kernels[1], padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(filters[1], conv_kernels[1], padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rates[0]))

    #Third convolutional block
    model.add(Conv2D(filters[2],conv_kernels[2], padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(filters[2], conv_kernels[2], padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rates[0]))
    #Third convolutional block
    model.add(Conv2D(filters[3],conv_kernels[3], padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(filters[3], conv_kernels[3], padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rates[0]))

    # Flatten and dense layers
    model.add(GlobalAveragePooling2D())

    model.add(Dense(neurons[0]))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rates[1]))
    
    model.add(Dense(neurons[1]))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rates[1]))

    model.add(Dense(neurons[1]))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rates[1]))

    # Output layer
    model.add(Dense(n_output, activation="softmax"))
    

    adam = Adam(learning_rate=0.001, beta_1=beta[0], beta_2=beta[1], epsilon=1e-07, decay=weight_decay)
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

    return model

#########################Loading and Augmenting Data#######################
image_dir = "images"
train_dir = os.path.join(image_dir, "train")
test_dir  = os.path.join(image_dir, "validation")


image_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    horizontal_flip=True,
    width_shift_range= .1,
    height_shift_range= .1,
    validation_split = .2
)

train_generator = image_data_generator.flow_from_directory(
    train_dir,
    target_size=input_shape[:-1],
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical',
    color_mode='grayscale',
    subset = "training"
)

val_generator = image_data_generator.flow_from_directory(
    train_dir,
    target_size=input_shape[:-1],
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical',
    color_mode='grayscale',
    subset = "validation"
)

test_generator = ImageDataGenerator(rescale= 1. / 255).flow_from_directory(
    test_dir,
    target_size=input_shape[:-1],
    batch_size=batch_size,
    shuffle=False,
    color_mode = "grayscale",
    class_mode='categorical'
)
#######################Builinding and Evaluating Model############################
model = build_model(filters, conv_kernels, dropout, neurons, weight_decay, beta)
model.summary()
hist = model.fit(train_generator, epochs=n_epoch, callbacks = [earlystop], validation_data=val_generator, verbose=1)

model.save("model_1")

result = model.evaluate(test_generator)

print("test set accuracy: ", result[1] * 100 , "%")

y_test = test_generator.classes
y_pred_prob = model.predict(test_generator)
y_pred = np.argmax(y_pred_prob, axis=1)
cnf_matrix = confusion_matrix(y_test, y_pred)

print(cnf_matrix)

###################Plotting Model History#####################

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.show()