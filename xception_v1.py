# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import confusion_matrix,classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, Xception, InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight


#%%
def mk_dir(path):
    try:
        os.makedirs(path)
    except:
        pass
#%% 
for run in [2, 3,4]:   
    path = os.getcwd()
    run_name = 'run_' +str(run)
    path = os.path.join(path, run_name)
    train_dataset_path = os.path.join(path, 'train')
    valid_dataset_path = os.path.join(path,'val')
    test_dataset_path = os.path.join(path, 'test')
    
    model_path =  os.path.join(path, 'model')
    out_path = os.path.join(path, 'out')
    mk_dir(out_path)
    mk_dir(model_path)
    #%%
    SAVE_MODEL = True
    TRAIN_MODEL = True
    
    BATCH_SIZE = 16
    IMG_WIDTH = 480
    IMG_HEIGHT = 848
    IMG_CHANNELS = 3
    IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Shape: {IMG_SHAPE}")
    
    #%%
    train_datagen = ImageDataGenerator(vertical_flip = True, # rescale = 1.0/ 255
                                      fill_mode = 'nearest')
    
    train_generator = train_datagen.flow_from_directory(train_dataset_path,
                                                       target_size = (IMG_WIDTH, IMG_HEIGHT),
                                                       batch_size = BATCH_SIZE,
                                                       class_mode = 'categorical',
                                                       shuffle = True,                                                   
                                                       seed = 2)
    
    
    test_datagen = ImageDataGenerator() #rescale = 1.0/ 255
    
    valid_generator = test_datagen.flow_from_directory(valid_dataset_path,
                                                     target_size = (IMG_WIDTH, IMG_HEIGHT),
                                                     batch_size = BATCH_SIZE,
                                                     class_mode = 'categorical',
                                                     shuffle = False)
    
    
   
    test_generator = test_datagen.flow_from_directory(test_dataset_path,
                                                     target_size = (IMG_WIDTH, IMG_HEIGHT),
                                                     batch_size = BATCH_SIZE,
                                                     class_mode = 'categorical',
                                                     shuffle = False)
    
    
    labels = {value: key for key, value in train_generator.class_indices.items()}
    
    print("Label Mappings for classes present in the training and validation datasets\n")
    for key, value in labels.items():
        print(f"{key} : {value}")
        
    # %%
    count = {}
    for key, value in labels.items():
        cnt = len(os.listdir(os.path.join(train_dataset_path, value)))
        count[key] = cnt
    print(count)
    max_count = max(count.values())
    count2 = {}
    for key in count.keys():
        count2[key]=1/(count[key]/max_count)
    print(count2)
    #%%
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
    idx = 0
    plt.suptitle("Sample Training Images", fontsize=20)
    for i in range(3):
        for j in range(3):
            label = labels[train_generator[0][1][idx].argmax()]
            ax[i, j].set_title(f"{label}")
            ax[i, j].imshow(train_generator[0][0][idx][:, :, :])
            ax[i, j].axis("off")
            idx += 1
    
    plt.tight_layout()
    # plt.show()
    #%%
    base_model = Xception(input_shape = IMG_SHAPE, include_top=False, weights='imagenet')
    
    base_model.trainable = False
    
    inputs = Input(shape = IMG_SHAPE)
    
    x = base_model(inputs, training = False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(3, activation='softmax')(x)
    
    model = Model(inputs = inputs, outputs = outputs)
    
    tf.keras.utils.plot_model(model)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                factor=np.sqrt(0.1),
                                 patience=5)
    
    base_learning_rate = 0.001
    optimizer = Adam(learning_rate=base_learning_rate)
    
    # Compute class weights to be used in the loss function
    # Compute class weights
    class_weights= count2
    
    # Compile your model with categorical cross-entropy loss
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    # Evaluate the model on the test set and print the initial metrics
    loss_0, accuracy_0 = model.evaluate(test_generator)
    with open(os.path.join(out_path, "output.txt"), "a") as f:
        f.write(f"Initial Loss: {loss_0:.2f}\n")
        f.write(f"Initial Accuracy: {accuracy_0:.2f}\n")
    
    if TRAIN_MODEL:
    
        initial_epochs = 100
        
        history = model.fit(
            train_generator,
            epochs=initial_epochs,
            validation_data=valid_generator,
            class_weight=class_weights,
            callbacks=[reduce_lr]
        )
        
        
        train_accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        learning_rate = history.history['lr']
        
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))
        
        ax[0].set_title('Training Accuracy vs. Epochs')
        ax[0].plot(train_accuracy, 'o-', label='Train Accuracy')
        ax[0].plot(val_accuracy, 'o-', label='Validation Accuracy')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend(loc='best')
        
        ax[1].set_title('Training/Validation Loss vs. Epochs')
        ax[1].plot(train_loss, 'o-', label='Train Loss')
        ax[1].plot(val_loss, 'o-', label='Validation Loss')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss')
        ax[1].legend(loc='best')
        
        ax[2].set_title('Learning Rate vs. Epochs')
        ax[2].plot(learning_rate, 'o-', label='Learning Rate')
        ax[2].set_xlabel('Epochs')
        ax[2].set_ylabel('Loss')
        ax[2].legend(loc='best')
        
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(out_path, "plot.png"))
        
    else:
    
        model = tf.keras.models.load_model(out_path)
    
    predictions = model.predict(test_generator)
    
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
    idx = 0
    plt.suptitle('Test Dataset Predictions', fontsize=20)
    
    for i in range(3):
        for j in range(3):
            predicted_labels = labels[np.argmax(predictions[idx])]
            ax[i, j].set_title(f"{predicted_labels}")
            ax[i, j].imshow(test_generator[0][0][idx])
            ax[i, j].axis("off")
            idx += 1
    
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(out_path, "plot0.png"))
    
    test_loss, test_accuracy = model.evaluate(test_generator)
    with open(os.path.join(out_path, "output.txt"), "a") as f:
        f.write(f"Test Loss:     {test_loss}\n")
        f.write(f"Test Accuracy: {test_accuracy}\n")
    
    
    y_pred = np.argmax(predictions, axis=1)  # change from np.round to np.argmax
    y_true = test_generator.classes
    
    cf_mtx = confusion_matrix(y_true, y_pred)
    
    group_counts = ["{0:0.0f}".format(value) for value in cf_mtx.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_mtx.flatten()/np.sum(cf_mtx)]
    box_labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(3, 3)  # change to 3 x 3
    
    plt.figure(figsize = (6, 6))
    sns.heatmap(cf_mtx, xticklabels=labels.values(), yticklabels=labels.values(),
               cmap="YlGnBu", fmt="", annot=box_labels)
    
    plt.xlabel('Predicted Classes')
    plt.ylabel('True Classes')
    plt.savefig(os.path.join(out_path, "plot1.png"))
    
    with open(os.path.join(out_path, "output.txt"), "a") as f:
        f.write("\n")
        f.write(classification_report(y_true, y_pred, target_names=labels.values()))
        f.write("\n")
    
    errors = (y_true - y_pred != 0)
    y_true_errors = y_true[errors]
    y_pred_errors = y_pred[errors]
    
    test_images = test_generator.filenames
    test_img = np.asarray(test_images)[errors]
    
    
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12, 10))
    idx = 0
    
    for i in range(2):
        for j in range(4):
            idx = np.random.randint(0, len(test_img))
            true_index = y_true_errors[idx]
            true_label = labels[true_index]
            predicted_index = y_pred_errors[idx]
            predicted_label = labels[predicted_index]
            ax[i, j].set_title(f"True Label: {true_label} \n Predicted Label: {predicted_label}")
            img_path = os.path.join(test_dataset_path, test_img[idx])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax[i, j].imshow(img)
            ax[i, j].axis("off")
    
    plt.tight_layout()
    plt.suptitle('Wrong Predictions made on test set', fontsize=20)
    plt.show()
    plt.savefig(os.path.join(out_path, "plot2.png"))
    
    #%%
    pred_df = pd.DataFrame(index = range(len(test_images)), columns = ['filename', "p_empty", "p_bad", "p_good", "pred", "true"])
    pred_df["filename"] = [im.split("/")[1] for im in test_images]
    pred_df[["p_empty", "p_bad", "p_good"]] = predictions
    pred_df["pred"] = [labels[p] for p in y_pred]
    pred_df["true"] = [labels[p] for p in y_true]
    
    pred_df.to_csv(os.path.join(out_path, "pred_df.csv"))
    #%%
    
    if SAVE_MODEL:
        model.save(model_path)
    
    else:
        print('Done')




