import os
import numpy as np
import keras
import tensorflow as tf
from skimage.io import imshow, imsave
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from plotly.offline import init_notebook_mode
import matplotlib.pyplot as plt
import scipy.io as sio
import spectral
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling3D, Conv3DTranspose, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


#----
# Save the loss and score function

class LossAndScoreRecorder(Callback):
    def __init__(self, log_file_path):
        super().__init__()
        self.log_file_path = log_file_path
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        epoch_logs = {
            'epoch': epoch + 1,
            'loss': logs['loss'],
            'mean_io_u': logs['mean_io_u'],
            'val_loss': logs['val_loss'],
            'val_mean_io_u': logs['val_mean_io_u']
        }
        self.logs.append(epoch_logs)
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(f"Epoch {epoch + 1}: Loss={epoch_logs['loss']:.4f}, Mean IoU={epoch_logs['mean_io_u']:.4f}, "
                           f"Validation Loss={epoch_logs['val_loss']:.4f}, Validation Mean IoU={epoch_logs['val_mean_io_u']:.4f}\n")

#----
n_classes = 4
def get_model():
    inputs = Input((144, 144, 30, 1))
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    # conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1) #84

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    # conv2 = BatchNormalization()(conv2)
    # conv2 = Dropout(0.1)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2) #42

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    # conv3 = BatchNormalization()(conv3)
    # conv3 = Dropout(0.1)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3) #21

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    # conv4 = BatchNormalization()(conv4)
    # conv4 = Dropout(0.1)(conv4)
    # pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4) #7

    # conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    # conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)
#     conv5 = BatchNormalization()(conv5)
#     conv5 = Dropout(0.2)(conv5)
    #pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

#     conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
#     conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)

    flat6 = Flatten()(conv4)
    output_1 = Dense(n_classes, activation='softmax', name='output_1')(flat6)

#     up7 = concatenate([Conv2DTranspose(512, (2, 2), padding='same')(conv6), conv5], axis=3)
#     conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(up7)
#     conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv7)

    # up8 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), conv4], axis=3)
    # conv8 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up8)
    # conv8 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv8)
#     conv8 = BatchNormalization()(conv8)
#     conv8 = Dropout(0.2)(conv8)

    up9 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=3)
    conv9 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv9)
    # conv9 = BatchNormalization()(conv9)
    # conv9 = Dropout(0.1)(conv9)

    up10 = concatenate([Conv3DTranspose(64, (2, 2,2 ), strides=(2, 2, 2), padding='same')(conv9), conv2], axis=3)
    conv10 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up10)
    conv10 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv10)
    # conv10 = BatchNormalization()(conv10)
    # conv10 = Dropout(0.1)(conv10)

    up11 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2,2, 2), padding='same')(conv10), conv1], axis=3)
    conv11 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up11)
    conv11 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv11)

    conv3d_shape = conv11.shape

    conv11 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv11)

    output_2 = Conv2D(n_classes, (1, 1), activation='softmax', name='output_2')(conv11)

    model = Model(inputs=[inputs], outputs=[ output_2])

    model.compile(optimizer=Adam(learning_rate=0.0005), loss={'output_2': 'categorical_crossentropy'}, metrics=[tf.keras.metrics.MeanIoU(num_classes=n_classes)])

    return model

#----
data_folder = '/home/student4/HSI/image2'
mask_folder = '/home/student4/HSI/mask2'
checkpoint_folder = '/home/student4/HSI/checkpoints5'
log_folder = '/home/student4/HSI/logs5'


# Get the list of file names in the data folder
x_file_names = [file for file in os.listdir(data_folder) if not file.startswith('.ipynb_checkpoints')]
y_file_names = [file for file in os.listdir(mask_folder) if not file.startswith('.ipynb_checkpoints')]

x_file_names.sort()
y_file_names.sort()

# Split data into training and validation sets
x_train_files, x_test_files, y_train_files, y_test_files = train_test_split(x_file_names, y_file_names, test_size=0.17, random_state=42)

#----

#Define the data generator function for loading.npy files
def data_generator(x_filenames, y_file_names, batch_size=1):
    while True:
        for i in range(0, len(x_filenames), batch_size):
            batch_x_filenames = x_filenames[i:i+batch_size]
            batch_y_filenames = y_file_names[i:i+batch_size]

            batch_x = []
            batch_y = []

            for x_file_name, y_file_name in zip(batch_x_filenames, batch_y_filenames):
                # Load the data and mask
                x = np.load(os.path.join(data_folder, x_file_name))
                y = np.load(os.path.join(mask_folder, y_file_name))
                y = tf.keras.utils.to_categorical(y, num_classes=4)

                x = x.reshape((1, 144, 144, 30, 1))
                y = y.reshape((1, 144, 144, n_classes))

                batch_x.append(x)
                batch_y.append(y)

            batch_x = np.concatenate(batch_x, axis=0)
            batch_y = np.concatenate(batch_y, axis=0)

            yield batch_x, {'output_2': batch_y}

#----
# Training data generator
train_data_generator = data_generator(x_train_files, y_train_files)

# Validation data generator
val_data_generator = data_generator(x_val_files, y_val_files)  

# Log file for recording loss and scores
log_file_path = os.path.join(log_folder, 'log3.txt')

# Create folders for saving checkpoints and logs
os.makedirs(checkpoint_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)

#----
model = get_model()
model.summary()
# Checkpoint
checkpoint_path = os.path.join(checkpoint_folder, 'checkpoint_{epoch:02d}_.h5')
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=False, mode='min', verbose=1)

# Tensorboard
tensorboard_callback = TensorBoard(log_dir=log_folder, histogram_freq=1)

# Calculate steps_per_epoch and validation_steps
steps_per_epoch = len(x_train_files)
validation_steps = len(x_val_files)

loss_and_score_recorder = LossAndScoreRecorder(log_file_path)
# Fit model
model.fit_generator(generator=train_data_generator, steps_per_epoch=steps_per_epoch, epochs=200,
                    callbacks=[checkpoint, tensorboard_callback, loss_and_score_recorder],
                    validation_data=val_data_generator, validation_steps=validation_steps)

#----
#Perform testing
# Load the trained model
model_path = '/home/student4/HSI/checkpoint5'
# Change to the path where your model is saved
model = load_model(model_path)

# Define the folder paths for testing data and labels
test_data_folder = '/home/student4/HSI/test'
test_mask_folder = '/home/student4/HSI/test_mask'

# Get the list of file names in the testing data folder
test_x_file_names = os.listdir(test_data_folder)
test_y_file_names = os.listdir(test_mask_folder)
test_x_file_names.sort()
test_y_file_names.sort()
assert len(test_x_file_names) == len(test_y_file_names), "Number of files in x_test and y_test folders do not match"

# Define the number of classes
n_classes = 4

# Function to preprocess and make predictions on a single test sample
def predict_sample(model, data_path):
    # Load the test data
    x_test = np.load(data_path)

    # Reshape the data to match the model input shape
    x_test = x_test.reshape(1, 144, 144, 30, 1)

    # Make predictions
    predictions = model.predict(x_test)

    # Return the predicted mask
    return predictions[0]

# Visualize the predictions
for test_x_file_name, test_y_file_name in zip(test_x_file_names, test_y_file_names):
    # Load the test data and ground truth mask
    test_data_path = os.path.join(test_data_folder, test_x_file_name)
    test_mask_path = os.path.join(test_mask_folder, test_y_file_name)

    x_test = np.load(test_data_path)
    y_test = np.load(test_mask_path)
    #y_test = np.argmax(y_test, axis=-1)
    # Reshape the data to match the model input shape
    x_test = x_test.reshape(1, 144, 144, 30, 1)

    # Make predictions
    predicted_mask = predict_sample(model, test_data_path)

    # Visualize the results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(x_test[0, :, :, 15, 0])  # Display one slice of the input data
    plt.title(f'Original Image_{test_x_file_name}')

    plt.subplot(1, 3, 2)
    plt.imshow(np.argmax(predicted_mask, axis=-1))  # Display the predicted mask
    plt.title(f'Predicted Mask_{test_y_file_name}')

    plt.subplot(1, 3, 3)
    plt.imshow(y_test)  # Display the ground truth mask
    plt.title('Ground Truth Mask')

    plt.show()
