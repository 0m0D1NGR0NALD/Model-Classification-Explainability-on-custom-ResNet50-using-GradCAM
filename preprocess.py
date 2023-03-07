import tensorflow as tf
import numpy as np

# Initialize target size
target_size = (512,512)

def img_array(image_path,target_size):
    # Load image and set target size
    img = tf.keras.preprocessing.image.load_img(image_path,target_size=target_size)
    # Convert image to array
    image_array = tf.keras.preprocessing.image.img_to_array(img)
    # Add a dimension to transform array into a batch
    array = np.expand_dims(image_array,axis=0)
    # Return array
    return array
