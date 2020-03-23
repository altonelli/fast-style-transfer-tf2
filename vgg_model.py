from collections import namedtuple
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model

VGG_Output = namedtuple('VGG_Output', 'content_output style_output')

class VGGModel:
  def __init__(self,
               content_layers=["conv4_2"],
               style_layers=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
               ):
    self.vgg = VGG19(include_top=False, weights='imagenet')
    self.layers = {
      "input"  : 0,
      "conv1_1": 1,
      "conv1_2": 2,
      "pool1"  : 3,
      "conv2_1": 4,
      "conv2_2": 5,
      "pool2"  : 6,
      "conv3_1": 7,
      "conv3_2": 8,
      "conv3_3": 9,
      "conv3_4": 10,
      "pool3"  : 11,
      "conv4_1": 12,
      "conv4_2": 13,
      "conv4_3": 14,
      "conv4_4": 15,
      "pool4"  : 16,
      "conv5_1": 17,
      "conv5_2": 18,
      "conv5_3": 19,
      "conv5_4": 20,
      "pool5"  : 21,
      "flatten": 22,
      "fc1"    : 23,
      "fc2"    : 24,
      "predictions": 25,
    }
    self.content_layers = content_layers
    self.style_layers = style_layers
    self.total_output_layers = self.content_layers + self.style_layers
    self.partition_idx = len(self.content_layers)
    self.model = Model(self.vgg.inputs, self._get_outputs(), trainable=False)

  def forward(self, X):
    outputs = self.model(X)
    return VGG_Output(outputs[:self.partition_idx], outputs[self.partition_idx:])

  def _get_outputs(self):
    return [self.vgg.layers[self.layers[layer]].output for layer in self.total_output_layers]

  def preprocess(self, images):
    images = tf.keras.applications.vgg19.preprocess_input(images)
    images = tf.image.resize(images, (224, 224))
    images = tf.cast(images, tf.float32)
    return images