import tensorflow as tf

def gradcam_heatmap(image_array,model,last_conv_layer_name,pred_index=None):
    # Create model that maps the input image to the activations of last conv layer & output predictions
    grad_model = tf.keras.models.Model(
        [model.get_layer(model.inputs).input],
        [model.get_layer(last_conv_layer_name).output,
         model.output])
    # Compute the gradient of the top predicted class for our image with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output,preds = grad_model(image_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:,pred_index]
    # Gradient of the output neuron with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel,last_conv_layer_output)
    # Mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads,axis=(0,1,2))
    # Multiply each channel in feature map array based on the top predicted class and sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output@pooled_grads[...,tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    # Normalize heatmap to [0 1] scale for better visualization
    heatmap = tf.maximum(heatmap,0)/tf.math.reduce_max(heatmap)
    # Return heatmap as numpy array
    return heatmap.numpy()