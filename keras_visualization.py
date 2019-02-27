from vgg16 import * 
from keras.utils import plot_model

model = VGG16(include_top=True, weights='imagenet') 

# Prints a summary representation of  model.
model.summary()

# Export model to .ps file
plot_model(model, to_file='model.ps', show_shapes=True)
