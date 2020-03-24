from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Conv2DTranspose, Layer
from tensorflow_addons.layers import InstanceNormalization


class ConvLayer(Layer):
  def __init__(self, filters,
               kernel=(3,3), padding='same',
               strides=(1,1), activate=True, name="",
               weight_initializer="glorot_uniform"
               ):
    super(ConvLayer, self).__init__()
    self.activate = activate
    self.conv = Conv2D(filters, kernel_size=kernel,
                       padding=padding, strides=strides,
                       name=name, trainable=True,
                       use_bias=False,
                       kernel_initializer=weight_initializer)
    self.inst_norm = InstanceNormalization(axis=3,
                                          center=True,
                                          scale=True,
                                          beta_initializer="zeros",
                                          gamma_initializer="ones",
                                          trainable=True)
    if self.activate:
      self.relu_layer = Activation('relu', trainable=False)

  def call(self, x):
    x = self.conv(x)
    x = self.inst_norm(x)
    if self.activate:
      x = self.relu_layer(x)
    return x


class ResBlock(Layer):
  def __init__(self, filters, kernel=(3,3), padding='same', weight_initializer="glorot_uniform", prefix=""):
    super(ResBlock, self).__init__()
    self.prefix_name = prefix + "_"
    self.conv1 = ConvLayer(filters=filters,
                           kernel=kernel,
                           padding=padding,
                           weight_initializer=weight_initializer,
                           name=self.prefix_name + "conv_1")
    self.conv2 = ConvLayer(filters=filters,
                           kernel=kernel,
                           padding=padding,
                           activate=False,
                           weight_initializer=weight_initializer,
                           name=self.prefix_name + "conv_2")
    self.add = Add(name=self.prefix_name + "add")

  def call(self, x):
    tmp = self.conv1(x)
    c = self.conv2(tmp)
    return self.add([x, c])


class ConvTLayer(Layer):
  def __init__(self, filters, kernel=(3,3), padding='same', strides=(1,1), activate=True, name="",
               weight_initializer="glorot_uniform"
               ):
    super(ConvTLayer, self).__init__()
    self.activate = activate
    self.conv_t = Conv2DTranspose(filters, kernel_size=kernel, padding=padding,
                                  strides=strides, name=name,
                                  use_bias=False,
                                  kernel_initializer=weight_initializer)
    self.inst_norm = InstanceNormalization(axis=3,
                                          center=True,
                                          scale=True,
                                          beta_initializer="zeros",
                                          gamma_initializer="ones",
                                          trainable=True)
    if self.activate:
      self.relu_layer = Activation('relu')

  def call(self, x):
    x = self.conv_t(x)
    x = self.inst_norm(x)
    if self.activate:
      x = self.relu_layer(x)
    return x
