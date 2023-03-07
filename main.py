import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from gradcam import gradcam_heatmap
from display_gradcam import save_and_display_gradcam
from preprocess import target_size

# Load model
model = tf.keras.models.load_model('resnet_50.h5')
# Initialize image path
image_path = "Data/covid.jpg"
# Convert image to array
image_array = img_array(image_path,target_size=target_size)
# Remove softmax activation in last layer
model.layers[-1].activation = None
# Make model prediction
prob = model.predict(image_array)
pred = np.argmax(prob)
# Generate class activation heatmap
heatmap = gradcam_heatmap(image_array,model,last_conv_layer_name="conv5_block3_out")
# Display heatmap
plt.matshow(heatmap,cmap='OrRd')
plt.colorbar()
plt.show()

# Dispay superimposed image
save_and_display_gradcam(image_path,heatmap)

# Display subplot of original image and superimposed image
plt.figure(figsize=(20,20),facecolor='black')
# Original image
plt.subplot(1,2,1)
img = plt.imread(image_path)
plt.imshow(img)
plt.grid(False)
plt.axis('off')
# Superimposed image
plt.subplot(1,2,2)
img_ = plt.imread(gradcam_path)
plt.imshow(img_)
plt.grid(False)
plt.axis('off')

plt.show()
