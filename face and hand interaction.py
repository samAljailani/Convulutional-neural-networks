from tensorflow.keras.models              import Sequential
from tensorflow.keras.layers              import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.callbacks           import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers        import l2
from tensorflow.keras.optimizers          import Adam
from sklearn.metrics                      import confusion_matrix, accuracy_score
from sklearn.model_selection              import StratifiedKFold
import cv2

import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import os
import itertools


###################################################################
input_shape    = (96, 96,3)
filters       = [ 32, 64, 128, 256]
kernels       = [(3,3), (3,3), (3,3), (3,3)]
learning_rate = 0.0001
dropout_rate  = 0.5
beta_1        = 0.9
beta_2        = 0.99
lambda_val    = 0.001
batch_size    = 32
n_output      = 3
n_epochs      = 70

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

def gaussian_blur(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image

###########Model Architecture#######################################
def build_model(input_shape, n_output, filters, kernels, dropout_rate, lr, beta_1, beta_2, lambda_val):
    model = Sequential()

    model.add(Conv2D(filters[0], kernels[0], padding="same", kernel_regularizer=l2(lambda_val),  input_shape=input_shape))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(filters[1], kernels[1], kernel_regularizer=l2(lambda_val), padding="same"))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(filters[2], kernels[2],kernel_regularizer=l2(lambda_val),  padding="same"))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(filters[3], kernels[3], kernel_regularizer=l2(lambda_val),  padding="same"))
    model.add(Activation('relu'))

    model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(128, kernel_regularizer=l2(lambda_val)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, kernel_regularizer=l2(lambda_val)))
    model.add(Activation('relu'))
    
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_output, activation='softmax'))

    optimizer = Adam(
                        learning_rate = lr,
                        beta_1=beta_1, 
                        beta_2=beta_2, 
                        epsilon=1e-07
                    )
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy']
    )
    return model



###############Loading Images and Data Augmentation####################
image_dir = "images"
train_dir = os.path.join(image_dir, "train")
val_dir = os.path.join(image_dir, "validation")
test_dir = os.path.join(image_dir, "test")
image_data_generator = ImageDataGenerator(
    preprocessing_function=gaussian_blur,
    rescale=1. / 255,
    rotation_range=25,
    horizontal_flip=True,
    width_shift_range= .1,
    height_shift_range= .1, 
    brightness_range=[0.5, 1.5],
    zoom_range= .2,
    shear_range=5
)

train_generator = image_data_generator.flow_from_directory(
    train_dir,
    target_size=input_shape[:-1],
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical'
)

val_generator = ImageDataGenerator(rescale= 1. / 255).flow_from_directory(
    val_dir,
    target_size=input_shape[:-1],
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical'

)

test_generator = ImageDataGenerator(rescale= 1. / 255).flow_from_directory(
    test_dir,
    target_size=input_shape[:-1],
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical'
)
#############Builind and Evaluating Model######################################

model = build_model(n_output = n_output,
                    input_shape= input_shape,
                    filters=filters,
                    kernels=kernels,
                    lr=learning_rate, 
                    dropout_rate= dropout_rate,
                    beta_1 = beta_1,
                    beta_2 = beta_2,
                    lambda_val=lambda_val
        )

model.summary()

hist = model.fit(train_generator,
                    epochs=n_epochs,  
                    validation_data=test_generator
                    ,
                    callbacks=[early_stopping]
                )
model.save("model")

result = model.evaluate(val_generator)

print("test set accuracy: ", result[1] * 100 , "%")

y_test = test_generator.classes
y_pred_prob = model.predict(test_generator)
y_pred = np.argmax(y_pred_prob, axis=1)
cnf_matrix = confusion_matrix(y_test, y_pred)

print(cnf_matrix)

##############Plotting Model history###########################
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
#####################Randomized Search CV Algorithm#####
#output: the dataframe of image paths in the given directory and their respective labels
def get_files_df(dir):
    classes = os.listdir(dir)
    files = []
    labels = []
    for label in classes:
        class_path = os.path.join(dir, label)
        for file in os.listdir(class_path): 
            file_path = os.path.join(class_path, file)
            files.append(file_path)
            labels.append(label)
    df = pd.DataFrame({"filename": files, "class": labels})
    return df
def randomized_search_cv(configs, k , n_iters):
    # Create the cross product of the configs
    all_configs = np.array(list(itertools.product(*configs.values())))

    n_iters = np.minimum(len(all_configs), n_iters)

    print("n_iters: ", n_iters)
    # Randomly select n_iters of combinations
    config_index = np.random.choice(len(all_configs), size=n_iters, replace=False)

    print("total conbinations of configutations: ", len(all_configs), "selecting: ", n_iters)
    df = get_files_df(train_dir)
    kfold = StratifiedKFold(n_splits=k, shuffle=True)
    config_scores = {}
    for i, config in enumerate(all_configs[config_index]): 
        cv_scores = []
        cv_histories = []
        cv_val_scores = []

        model = build_model(n_output = n_output,
                                input_shape= config[0],
                                filters=config[1],
                                kernels=config[2],
                                lr=config[3], 
                                dropout_rate= config[4],
                                beta_1 = config[5],
                                beta_2 = config[6],
                                lambda_val=config[7]
                    )
        for j, (train_index, val_index) in enumerate(kfold.split(df['filename'], df['class'])):
            print("\n===============================================\n")
            print("config #", i + 1, "fold # ", j + 1)
            print("config:", config )
            train_data = df.iloc[train_index]
            val_data = df.iloc[val_index]

            train_generator = image_data_generator.flow_from_dataframe(
                train_data,
                x_col='filename',
                y_col='class',
                target_size=config[0][:-1],
                batch_size=batch_size,
                shuffle=True,
                class_mode='categorical'
            )

            val_generator = image_data_generator.flow_from_dataframe(
                val_data,
                x_col='filename',
                y_col='class',
                target_size=config[0][:-1],
                batch_size=batch_size,
                shuffle=True,
                class_mode='categorical'
            )

            history = model.fit(train_generator,
                                epochs=n_epochs,  
                                validation_data=val_generator
                                ,
                                callbacks=[early_stopping]
                            )

            val_generator_no_aug = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
                val_data,
                x_col='filename',
                y_col='class',
                target_size=config[0][:-1],
                batch_size=batch_size,
                shuffle=False,
                class_mode='categorical'
            )

            test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
                val_dir,
                target_size=config[0][:-1],
                batch_size=batch_size,
                shuffle=False,
                class_mode='categorical'
            )


            y_val = val_generator_no_aug.classes
            y_pred_prob = model.predict(val_generator_no_aug)
            y_pred = np.argmax(y_pred_prob, axis=1)

            accuracy = accuracy_score(y_val, y_pred)

            val_accuracy = accuracy_score(test_generator.classes, np.argmax(model.predict(test_generator), axis=1))
            print(f'Accuracy for this fold: {accuracy * 100:.2f}%')
            print(f'test set accuracy: ', val_accuracy)
            cv_scores.append(accuracy)
            cv_histories.append(history)

        ###########storing results in a file##################
        mean_val_accuracy = np.mean(cv_val_scores)
        mean_accuracy = np.mean(cv_scores)
        with open("kfold_accuracies.txt", 'a') as f:
            f.write(f'config: {config}')
            for i, accuracy in enumerate(cv_scores, 1):
                f.write(f'Accuracy for fold {i}: {accuracy * 100:.2f}%\n')
                f.write(f'valdation set accuracy for fold {i}: {cv_val_scores[i-1] * 100:.2f}%\n')
            f.write(f'Mean accuracy over {k}-fold cross-validation: {mean_accuracy * 100:.2f}%\n')
            f.write(f'Mean accuracy over validation set: {mean_val_accuracy * 100:.2f}%\n')
            f.write("\n==================================\n")
        print("CONFIG END:", config, "accuracy: ", mean_accuracy)
    # Plot the top 5 configs as box plots
    best_5_configs = sorted(config_scores, key=config_scores.get, reverse=True)[:5]
    best_5_accuracies = [config_scores[config] for config in best_5_configs]
    return best_5_configs, best_5_accuracies, cv_histories