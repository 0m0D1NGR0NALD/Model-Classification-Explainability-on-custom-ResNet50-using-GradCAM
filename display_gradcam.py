import tensorflow as tf
import numpy as np
from IPython.display import Image,display

def save_and_display_gradcam(image_path,heatmap,gradcam_path="gradcam.png",heatmap_path="heatmap.png",alpha=5):
    # Load the image
    img = tf.keras.preprocessing.image.load_img(image_path)
    # Convert image to array
    img = tf.keras.preprocessing.image.img_to_array(img)
    # Rescale heatmap to a range of [0 255]
    heatmap = np.uint8(255*heatmap)
    # Use jet colormap to colorize heatmap
    jet = cm.get_map("jet")
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:,:3]
    jet_heatmap = jet_colors[heatmap]
    # Create an image with RGB colorized heatmap
    # Convert array to image
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    # Resize image
    jet_heatmap = jet_heatmap.resize((img.shape[1],img.shape[0]))
    # Convert image to array
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap*alpha+img
    # Convert superimposed array to image
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    # Save the superimposed image
    superimposed_img.save(gradcam_path)
    # Covert heatmap array to image
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap*alpha)
    # Save the heatmap
    jet_heatmap.save(heatmap_path)
    # Display GradCAM
    display(Image(gradcam_path))
