#################################################
## AUTHOR: James Beasley                       ##
## DATE: February 9, 2017                      ##
## UDACITY SDC: Project 3 (Behavioral Cloning) ##
#################################################

#ensure we're in the correct directory before we import other things
import os
assert (os.path.split(os.getcwd())[1] == 'data'), 'model.py must be executed within the "data" directory!'

#############
## IMPORTS ##
#############
import cv2
import csv
import random
import numpy as np
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

######################################################
## LOAD TRAINING DATA (IMAGE-PATHS/STEERING-ANGLES) ##
######################################################
drive_log_image_paths = [] #file paths (left, center, and right images are all folded in to this vector of image paths)
drive_log_steering_angles = [] #steering angle associated with drive_log_image_paths
max_angle_offset = 0.25 #max offset that can be applied to left/right images
min_angle_offset = 0.2 #min offset that can be applied to left/right images
cur_steering_angle = 0.0 #steering angle from cur_row

#load the driving log data (folding in the left, center, and right images into one vector)
with open('driving_log.csv') as file_handle:
    #view file rows as a dictionary (based on header row)
    dict_reader = csv.DictReader(file_handle)
    #get current row as dictionary
    for cur_dict_line in dict_reader:
        #get steering angle from cur_row
        cur_steering_angle = float(cur_dict_line['steering']) 
        #get rid of zero degree angles
        if (cur_steering_angle != 0):
            #add the left, center, and right image paths from cur_row 
            drive_log_image_paths.append((cur_dict_line['left']).strip())
            drive_log_image_paths.append((cur_dict_line['center']).strip())
            drive_log_image_paths.append((cur_dict_line['right']).strip())
            #append steering angle + random angle offset (for left)
            drive_log_steering_angles.append(cur_steering_angle + np.random.uniform(low=min_angle_offset, high=max_angle_offset))
            #append center steering angle (no change)
            drive_log_steering_angles.append(cur_steering_angle)
            #append steering angle - random angle offset (for right)
            drive_log_steering_angles.append(cur_steering_angle - np.random.uniform(low=min_angle_offset, high=max_angle_offset))
        
#convert to array
drive_log_steering_angles = np.array(drive_log_steering_angles)
drive_log_image_paths = np.array(drive_log_image_paths)

print()
print()
print("Displaying stats for modified Udacity-provided training data...")
print("Num drive log image paths:", len(drive_log_image_paths))
print("Num drive log steering angles:", len(drive_log_steering_angles))
print("Min drive log steering angle:", min(drive_log_steering_angles))
print("Max drive log steering angle:", max(drive_log_steering_angles))
print()
print()

#########################################################
## SHUFFLE TRAINING DATA (IMAGE-PATHS/STEERING-ANGLES) ##
#########################################################
drive_log_image_paths, drive_log_steering_angles = shuffle(drive_log_image_paths, drive_log_steering_angles)

##############################################################
## CARVE OUT A PORTION OF TRAINING SET FOR MODEL VALIDATION ##
##############################################################
X_train, X_validation, y_train, y_validation = train_test_split(drive_log_image_paths, drive_log_steering_angles, test_size=0.2, random_state=0)

print("Splitting training and validation data sets...")
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_validation shape: ", X_validation.shape)
print("y_validation shape: ", y_validation.shape)
print()
print()

######################
## DEFINE CNN MODEL ##
######################

#NVIDIA model + inline-cropping, relu, dropout
def cnn_model(dropout_keep_prob):
    #create sequential model (stacked layers)
    model = Sequential()
    #CROP: Input = 160x320x3, Output = 66x200x3
    model.add(Cropping2D(cropping=((70, 24), (60, 60)), input_shape=(160, 320, 3)))
    #NORMALIZE (-0.5 to 0.5): Input = 66x200x3, Output = 66x200x3
    model.add(Lambda(lambda x: (x - 127.5) / 255))
    ## LAYER 1 ##
    #CONVOLUTIONAL: Input = 66x200x3, Output = 31x98x24
    #filter: 5x5, input depth: 3, output depth: 24
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', activation="relu", init='lecun_uniform'))
    ## LAYER 2 ##
    #CONVOLUTIONAL: Input = 31x98x24, Output = 14x47x36
    #filter: 5x5, input depth: 24, output depth: 36
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation="relu", init='lecun_uniform'))
    ## LAYER 3 ##
    #CONVOLUTIONAL: Input = 14x47x36, Output = 5x22x48
    #filter: 5x5, input depth: 36, output depth: 48
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation="relu", init='lecun_uniform'))
    ## LAYER 4 ##
    #CONVOLUTIONAL: Input = 5x22x48, Output = 3x20x64
    #filter: 3x3, input depth: 48, output depth: 64
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation="relu", init='lecun_uniform'))
    ## LAYER 5 ##
    #CONVOLUTIONAL: Input = 3x20x64, Output = 1x18x64
    #filter: 3x3, input depth: 64, output depth: 64
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation="relu", init='lecun_uniform'))
    #FLATTEN INPUT WHILE RETAINING BATCH: Input = 1x18x64, Output = 1152x1
    model.add(Flatten())
    ## LAYER 6 ##
    #FULLY-CONNECTED: Input = 1152, Output = 100
    model.add(Dense(100, activation='relu', init='lecun_uniform'))
    ## LAYER 7 ##
    #FULLY-CONNECTED: Input = 100, Output = 50
    model.add(Dense(50, activation='relu', init='lecun_uniform'))
    #REGULARIZATION: Dropout
    model.add(Dropout(dropout_keep_prob))
    ## LAYER 8 ##
    #FULLY-CONNECTED: Input = 50, Output = 10
    model.add(Dense(10, activation='relu', init='lecun_uniform'))
    #REGULARIZATION: Dropout
    model.add(Dropout(dropout_keep_prob))
    ## OUTPUT LAYER ##
    #FULLY-CONNECTED: Input = 10, Output = 1
    model.add(Dense(1, init='lecun_uniform'))
    #return completed model
    return model

###################################################
## GENERATE VALIDATION & SYNTHETIC TRAINING DATA ##
###################################################

#translate (change position of) training example
#translation matrix found here: http://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html
def generate_translation_matrix(image, steering_angle):
    #randomly translate x
    translated_x = np.random.uniform(low=-15, high=15)
    #adjust steering angle based on x translation
    steering_angle += translated_x * 0.003
    #randomly translate y
    translated_y = np.random.uniform(low=-15, high=15)
    #return translation matrix based on above values
    return (np.float32([[1, 0, translated_x],[0, 1, translated_y]]), steering_angle)

#perform brightness adjustment (brighten or darken)
def perform_brightness_adjustment(image, steering_angle):
    #convert RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #randomly adjust V channel
    hsv[:, :, 2] = hsv[:, :, 2] * np.random.uniform(low=0.2, high=1.0)
    #convert back to RGB and return (steering angle is unaltered)
    return (cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), steering_angle)

#perform y-axis flip
def perform_y_axis_flip(image, steering_angle):
    #return y-axis flipped image and flipped steering angle
    return (np.fliplr(image), -steering_angle)

#randomly translate object (within specified matrix bounds)
def perform_translation(image, steering_angle):
    #get tanslation matrix along with adjusted steering angle
    object_transform_matrix, steering_angle = generate_translation_matrix(image, steering_angle)
    #return randomly translated image with adjusted steering angle
    return (cv2.warpAffine(image, object_transform_matrix, (image.shape[1], image.shape[0])), steering_angle)

#generate a synthetic example from the supplied training example
def generate_synthetic_training_example(image, steering_angle):
    #list of transformation functions available
    transformation_functions = [perform_translation, perform_brightness_adjustment, perform_y_axis_flip]
    #choose the number of transformations to perform at random (between 1 and 3)
    num_transformations_to_perform = random.randint(1, len(transformation_functions))
    #perform the number of transformations chosen
    for _ in range(0, num_transformations_to_perform):
        #select a transformation function at random
        selected_transformation_function = random.choice(transformation_functions)           
        #execute the transformation function and return the result
        image, steering_angle = selected_transformation_function(image, steering_angle)
        #ensure each transformation can only be performed once by removing it from the list
        transformation_functions.remove(selected_transformation_function)
    #return transformed image & adjusted steering angle
    return (image, steering_angle)

#on-the-fly synthetic data generator
def generate_synthetic_training_batch(X_train, y_train, batch_size):
    #loop forever
    while True:
        X_train_synthetic = []  #batch of images (after-processing)
        y_train_synthetic = []  #batch of steering angles (after-processing)
        #shuffle data
        X_train, y_train = shuffle(X_train, y_train)
        #create enough synthetic images to fill a batch
        for i in range(batch_size):
            #randomly select an index within X_train (zero indexed)
            random_index = np.random.randint(len(X_train))
            #load image
            image = mpimg.imread(X_train[random_index])
            #create a synthetic example based on that image
            synthetic_image, steering_angle = generate_synthetic_training_example(image, y_train[random_index])
            #append synthetic image
            X_train_synthetic.append(synthetic_image)
            #append steering angle
            y_train_synthetic.append(steering_angle)
        #yeild a new batch
        yield (np.array(X_train_synthetic), np.array(y_train_synthetic))

#generate batches of validation data
def generate_validation_batch(X_validation, y_validation, batch_size):
    #determine validation set length
    num_validation_examples = len(X_validation)
    #loop forever
    while True:
        #shuffle data
        X_validation, y_validation = shuffle(X_validation, y_validation)
        #walk through validation set loading image batches equal to batch size and yielding them
        for offset in range(0, num_validation_examples, batch_size):
            #list to store loaded images
            X_validation_image_batch = []
            #get the current batch of image paths
            cur_image_path_batch = X_validation[offset:offset+batch_size]
            #load images from paths contained in cur_image_path_batch 
            for image_path in cur_image_path_batch:
                #load image
                image = mpimg.imread(image_path)
                #append to validation image batch list
                X_validation_image_batch.append(image)
            #yeild batch
            yield (np.array(X_validation_image_batch), y_validation[offset:offset+batch_size])

#################
## TRAIN MODEL ##
#################

print("Begin training model...")
print()
print()

#get a handle to the model
model = cnn_model(dropout_keep_prob=0.5)    

#init the optimizer, loss function, and metrics for the model
model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

#init generators
#generates a synthetic batch based on the training data
training_generator = generate_synthetic_training_batch(X_train, y_train, batch_size=256)
#fetches a batch from the validation data we carved out earlier
validation_generator = generate_validation_batch(X_validation, y_validation, batch_size=256)

#define training callbacks
callbacks_list = [
    #stop if we're not improving
    EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min'),
    #save only the best model
    ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False, period=1)
]

#fit the model on batches of real-time synthetic data:
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    samples_per_epoch=25600,
                    nb_val_samples=5120,
                    nb_epoch=4,
                    verbose=2,
                    callbacks=callbacks_list)
