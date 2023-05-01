import numpy as np 
import pandas as pd 
import os
from tensorflow.keras.utils import load_img, img_to_array
from keras.utils import to_categorical
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping

# from concurrent import futures


def append_df(path, classes, num_classes):
    """
    creates two lists holding the path to images and the labels for respective images
    
    path (str): path to training/validation folder
    classes (list): labels to classify (directory names in path)
    num_classes: number of classes to classify
    """
    # lists to hold imagepath and labels
    file_names = []
    file_labels = []
    
    for labels in range(num_classes):
        temp_path = os.path.join(path, classes[labels])  # classes -> ['class1', 'class2']
        img_list = os.listdir(temp_path)  # temp_path -> image names in 'class1'
        num_img_files = len(img_list)  # get the number of images in the class

        # Get the path to each image
        for images in range(num_img_files):
            img_name = img_list[images]
            img_filename = os.path.join(temp_path, img_name)
            # append path to image and respective label
            file_names.append(img_filename)
            file_labels.append(labels)
    return [[file, label] for file, label in zip(file_names, file_labels)]


def pre_gen(path):
    """
    Get path to images and respective labels
    """
    classes = os.listdir(path)
    num_classes = len(classes)
    samples = append_df(path, classes, num_classes)
    return samples, len(samples), num_classes


def generator(path, batch_size=32, img_size=224, multi_output=True, augmentation=False):
    """
    Yield training samples in batches
    """
    samples, num_samples, num_classes = pre_gen(path)
    root = os.getcwd()
    while True:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            x = []
            y = []
            for batch_sample in batch_samples:
                img_name = batch_sample[0]
                label = batch_sample[1]
                img = load_img(os.path.join(root, img_name), target_size=(img_size, img_size))
                img = img_to_array(img)
                x.append(img)
                y.append(label)

            y = np.array(y)
            y = to_categorical(y, num_classes=num_classes)

            if multi_output:
                y = np.transpose(y)

                targets = {}
                for i in range(num_classes):
                    targets[f'model_{i}'] = y[i]  # targets -> {'model1': [0, 0, 1, 0, ...]}
            else:
                targets = y
            if augmentation:
                x = list(map(augmentation_func, x))  # multiprocessing isn't used since we are loading in small batches
            x = np.array(x)
            x = x/255.0
            yield x, targets


def augmentation_func(x):
    
    x = image.random_channel_shift(x, 100)
    x = image.random_shear(x, 40, row_axis=0, col_axis=1, channel_axis=2)
    x = image.random_rotation(x, 40, row_axis=0, col_axis=1, channel_axis=2)
    return x

            
def train_model(train_path, 
                val_path, 
                epochs=100, 
                multi_output=True, 
                batch_size=32, 
                blocks=[64, 32],
                patience=3,
                dropout=0,
                augmentation=False,
                verbose=1
                ):
    """
    Train VGG16 model on dataset using multiple prediction layers or single prediction layer
    """
    _, num_train_samples, num_classes = pre_gen(train_path)
    _, num_val_samples, _ = pre_gen(val_path)
    train_generator = generator(train_path, multi_output=multi_output, batch_size=batch_size, augmentation=augmentation)
    val_generator = generator(val_path, multi_output=multi_output, augmentation=False)
    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for block in blocks:
        x = Dense(block, activation='relu')(x)
        x = Dropout(dropout)(x)
 
    if multi_output:
        # outputs is a list holding the multiple binary classifiers. -> [model_0, model_1, ...]
        outputs = [Dense(1, activation='sigmoid', name=f'model_{i}')(x) for i in range(num_classes)]
        # get the loss for each binary classifier
        loss = {}
        for i in range(num_classes):
            loss[f'model_{i}'] = 'binary_crossentropy'
        # set callback
        callbacks = MonitorAvgAccuracy(patience=patience, num_classes=num_classes)
    else:
        outputs = Dense(num_classes, activation='softmax')(x)
        loss = 'categorical_crossentropy'
        callbacks = EarlyStopping(monitor='val_acc', patience=patience, restore_best_weights=True)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    model.compile(optimizer='adam', loss=loss, metrics=['acc'])

    # Train and validate model.
    history = model.fit(train_generator,
                        steps_per_epoch=num_train_samples // batch_size,
                        epochs=epochs,
                        callbacks=[callbacks],
                        verbose=verbose,
                        validation_data=val_generator,
                        validation_steps=num_val_samples // batch_size)
    return history.history, model


class MonitorAvgAccuracy(Callback):
    def __init__(self, num_classes, patience=5, save_best_weights=True):
        super().__init__()
        self.num_classes = num_classes
        self.patience = patience
        self.save_best_weights = save_best_weights
        self.best_avg_accuracy = -1.0
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.wait = 0
        self.stopped_epoch = 0
        self.best_avg_accuracy = -1.0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        sum_acc = 0
        for key, value in logs.items():
            if ('val' in key) and ('acc' in key):
                sum_acc += value
        avg_acc = sum_acc/self.num_classes
        # Add the average accuracy to the logs dictionary
        logs["val_avg_acc"] = avg_acc

        # Early stopping and best model saving
        if avg_acc > self.best_avg_accuracy:
            self.best_avg_accuracy = avg_acc
            self.wait = 0
            if self.save_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.save_best_weights:
                    self.model.set_weights(self.best_weights)
