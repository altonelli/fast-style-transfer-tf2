import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv2D, Add, Activation, Conv2DTranspose
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Layer


class ConvLayer(Layer):
  def __init__(self, filters, kernel=(3,3), padding='same', strides=(1,1),
               activate=True, weight_initializer="glorot_uniform", name=""):
    super(ConvLayer, self).__init__()
    self.conv = Conv2D(filters,
                       kernel_size=kernel,
                       padding=padding,
                       strides=strides,
                       weight_initializer=weight_initializer,
                       use_bias=False,
                       name=name)
    self.inst_norm = InstanceNormalization(axis=3,
                                           center=True,
                                           scale=True,
                                           beta_initializer="zeros",
                                           gamma_initializer="ones",
                                           trainable=False)
    self.relu_layer = Activation('relu')
    self.activate = activate

  def __call__(self, x):
    x = self.conv(x)
    x = self.inst_norm(x)
    if self.activate:
      x = self.relu_layer(x)
    return x


class ResBlock(Layer):
  def __init__(self, filters, kernel, padding='same', prefix=""):
    super(ResBlock, self).__init__()
    self.prefix_name = prefix + "_"
    self.conv1 = ConvLayer(filters=filters, kernel=kernel, padding=padding, name=self.prefix_name + "conv_1")
    self.conv2 = ConvLayer(filters=filters, kernel=kernel, padding=padding, activate=False, name=self.prefix_name + "conv_2")
    self.add = Add(name=self.prefix_name + "add")

  def __call__(self, x):
    tmp = self.conv1(x)
    c = self.conv2(tmp)
    return self.add([x, c])


class ConvTLayer(Layer):
  def __init__(self, filters, kernel=(3,3), padding='same', strides=(1,1), activate=True,
               weight_initializer="glorot_uniform", name=""):
    super(ConvTLayer, self).__init__()
    self.conv_t = Conv2DTranspose(filters,
                                  kernel_size=kernel,
                                  padding=padding,
                                  strides=strides,
                                  weight_initializer=weight_initializer,
                                  use_bias=False,
                                  name=name)
    self.inst_norm = InstanceNormalization(axis=3,
                                           center=True,
                                           scale=True,
                                           beta_initializer="zeros",
                                           gamma_initializer="ones",
                                           trainable=True)
    self.relu_layer = Activation('relu')
    self.activate = activate

  def __call__(self, x):
    x = self.conv_t(x)
    x = self.inst_norm(x)
    if self.activate:
      x = self.relu_layer(x)
    return x
