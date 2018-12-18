# import os
# os.environ["PATH"] += os.pathsep + 'D:/graphviz2.38/bin/'
#
from keras.models import Model, load_model
# from keras.utils import plot_model
#
#
model = load_model('model_post_augmentation.h5', compile = False)
# plot_model(model, to_file='model_visualization.png')

layers = model.layers
inputs = model.inputs
outputs = model.outputs
summary = model.summary()

test = 5