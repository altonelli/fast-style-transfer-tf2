import tensorflow as tf
from layers import ConvLayer, ConvTLayer, ResBlock
from tensorflow.keras.layers import Activation

class TransformNet:
  def __init__(self):
    self.conv1 = ConvLayer(32, (9,9), strides=(1,1), padding='same', name="conv_1")
    self.conv2 = ConvLayer(64, (3,3), strides=(2,2), padding='same', name="conv_2")
    self.conv3 = ConvLayer(128, (3,3), strides=(2,2), padding='same', name="conv_3")
    self.res1 = ResBlock(128, prefix="res_1")
    self.res2 = ResBlock(128, prefix="res_2")
    self.res3 = ResBlock(128, prefix="res_3")
    self.res4 = ResBlock(128, prefix="res_4")
    self.res5 = ResBlock(128, prefix="res_5")
    self.convt1 = ConvTLayer(64, (3,3), strides=(2,2), padding='same', name="conv_t_1")
    self.convt2 = ConvTLayer(32, (3,3), strides=(2,2), padding='same', name="conv_t_2")
    self.conv4 = ConvLayer(3, (9,9), strides=(1,1), padding='same', relu=False, name="conv_4")
    self.tanh = Activation('tanh')
    self.model = self._get_model()

  def _get_model(self):
    inputs = tf.keras.Input(shape=(None,None,3))
    x = self.conv1(inputs)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.res1(x)
    x = self.res2(x)
    x = self.res3(x)
    x = self.res4(x)
    x = self.res5(x)
    x = self.convt1(x)
    x = self.convt2(x)
    x = self.conv4(x)
    x = self.tanh(x)
    x = (x + 1) * (255. / 2)
    return tf.keras.Model(inputs, x, name="transformnet")

  def get_variables(self):
    return self.model.trainable_variables

  def preprocess(self, img):
    return img / 255.0

  def unprocess(self, img):
    return tf.clip_by_value(img, 0.0, 255.0)
