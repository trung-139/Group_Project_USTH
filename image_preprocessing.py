import os
import spectral as sp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split
from shutil import copyfile
#----
# Load the hyperspectral image
hdr_path = '/home/student4/HSI/Hyper-Spectral/hyper_20220326_3cm.hdr'
data_dir = '/home/student4/HSI/Hyper-Spectral/hyper_20220326_3cm.img'
os.environ['SPECTRAL_DATA'] = data_dir

img = sp.open_image(hdr_path) 
#----
# Load the label ground truth
label = Image.open('home/student4/HSI/label')
label_array = np.array(label)

#----
# The function of cropping the original image into a smaller image that contains most of the crop field information
def crop_and_stack_bands(img, crop_size=4320, start_x=None, start_y=None):
    # Get the number of bands
    num_bands = img.shape[2]

    # Initialize an empty list to store the cropped bands
    cropped_bands_list = []

    # Loop through each band
    for band in range(num_bands):
        # Read the current band
        current_band = img.read_band(band)  # Bands in Spectral start from 1

        # Set default starting points if not provided
        if start_x is None:
            start_x = current_band.shape[0] // 2 - crop_size // 2
        if start_y is None:
            start_y = current_band.shape[1] // 2 - crop_size // 2

        # Calculate the cropping boundaries
        end_x = start_x + crop_size
        end_y = start_y + crop_size

        # Crop the band
        cropped_band = current_band[start_x:end_x, start_y:end_y]

        # Append the cropped band to the list
        cropped_bands_list.append(cropped_band)

    # Convert the list of cropped bands to a NumPy array
    cropped_bands = np.stack(cropped_bands_list, axis=-1)
    return cropped_bands
#----

# PCA to reduce the dimension of the channels of the hyperspectral image
def applyPCA(X, numComponents=30):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca


# The image cropped started at pixel with the x = 1500, y = 1500 and have the shape of (4320,4320, )
cropped_array = crop_and_stack_bands(img, start_x=1500, start_y=1500)
cropped_label = crop_and_stack_bands(img, start_x=1500, start_y=1500)
#----
# The size of the input is (144,144,30)
# So we want to crop and PCA the image that is cropped above into patches and save them into .npy file
countimg = 0
for i in range (0, 59):
    for j in range (0,59):
        custom_start_x = 72*i
        custom_start_y = 72*j

        cropped = crop_and_stack_bands(cropped_array, start_x=custom_start_x, start_y=custom_start_y)
        pca_result, pca_model = applyPCA(cropped, 30)
        np.save(f'home/student4/HSI/image/image_{countimg}.npy', pca_result)
        countimg += 1


# Do the same with the mask
countlb = 0      
for a in range (0, 59):
    for b in range (0,59):
        custom_start_x = 72*i
        custom_start_y = 72*j

        cropped_lb = crop_and_stack_bands(cropped_label, start_x=custom_start_x, start_y=custom_start_y)
        pca_result, pca_model = applyPCA(cropped_lb, 30)
        np.save(f'home/student4/HSI/mask/mask_{countlb}.npy', pca_result)

#----
# Filter the images that their masks only contain background or not land (black color) to reduce computational
#resources and focus on the main task of the project

# Define paths
image_folder = "train"
mask_folder = "mask"
output_image_folder = "image2"
output_mask_folder = "mask2"

# Create output folders if they don't exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

# Loop through image files
for filename in os.listdir(image_folder):
    if filename.startswith("image_"):
        image_path = os.path.join(image_folder, filename)
        mask_filename = f"mask_{filename.split('_')[1]}"
        mask_path = os.path.join(mask_folder, mask_filename)

        # Load mask
        mask = np.load(mask_path)

        # Check if mask contains any non-zero values
        if np.any(mask != 0):
            # Copy image to output folder
            copyfile(image_path, os.path.join(output_image_folder, filename))

            # Copy mask to output folder
            copyfile(mask_path, os.path.join(output_mask_folder, mask_filename))
#----