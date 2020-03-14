from collections import namedtuple
from glob import glob
import os
import datetime
import csv
import numpy as np
import tensorflow as tf
from shutil import copy2
from distutils.dir_util import copy_tree

from transform_net import TransformNet
from vgg_model import VGGModel

Loss = namedtuple('Loss', 'total_loss style_loss content_loss tv_loss')

class Trainer:
    def __init__(self,
                 style_path,
                 content_file_path,
                 epochs,
                 batch_size,
                 content_weight,
                 style_weight,
                 tv_weight,
                 learning_rate,
                 log_dest_root,
                 save_dest_root,
                 log_period=100,
                 save_period=1000,
                 content_layers=("conv4_2",),
                 style_layers=("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"),
                 c_layer_weights=(1,),
                 s_layer_weights=(0.2, 0.2, 0.2, 0.2, 0.2)):
        self.style_path = style_path
        self.style_name = style_path.split("/")[-1].split(".")[0]
        self.content_file_path = content_file_path
        assert (len(content_layers) == len(c_layer_weights))
        self.content_layers = content_layers
        self.c_layer_weights = c_layer_weights
        assert (len(style_layers) == len(s_layer_weights))
        self.style_layers = style_layers
        self.s_layer_weights = s_layer_weights
        self.partition_idx = len(self.content_layers)
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_period = log_period
        t = datetime.datetime.today()
        self.train_date = "%d-%d-%d" % (t.year, t.month, t.day)
        self.log_file = "/tmp/log_file_" + self.train_date + ".csv"
        self.log_dest_path = os.path.join(log_dest_root, self.style_name, self.train_date)
        self.save_period = save_period
        self.saved_model_path = "/tmp/saved_models"
        self.save_dest_path = os.path.join(save_dest_root, self.style_name, self.train_date)

        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_weight = tv_weight

        self.transform = TransformNet()
        self.vgg = VGGModel(
            content_layers=content_layers,
            style_layers=style_layers,
            partition_idx=self.partition_idx
        )
        self.learing_rate = learning_rate
        self.train_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learing_rate)

    def run(self):
        self.A_outputs = self._get_A_base_outputs()
        self.A_style_grams = [self._gram_matrix(tf.convert_to_tensor(m, tf.float32)) for m in
                              self.A_outputs[self.vgg.partition_idx:]]

        content_images = glob(os.path.join(self.content_file_path, "*.jpg"))
        num_images = len(content_images) - (len(content_images) % self.batch_size)
        print(num_images)

        os.makedirs(self.log_dest_path, exist_ok=True)
        os.makedirs(self.save_dest_path, exist_ok=True)

        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "total_loss", "style_loss", "content_loss", "tv_loss"])

        for e in range(self.epochs):
            iter_list = [content_images[i:i + self.batch_size] for i in range(0, num_images, self.batch_size)]
            for e_i, batch in enumerate(iter_list):
                imgs = [ImageUtils.get_image(img_path, (256, 256, 3)) for img_path in batch]
                imgs = np.array(imgs)

                # Treat as X now that it is a tensor
                X = tf.convert_to_tensor(imgs)

                with tf.GradientTape() as tape:
                    L = self._forward_pass(X)

                # out of tape
                trainable_vars = self.transform.get_variables()
                grads = tape.gradient(L.total_loss, trainable_vars)

                self.train_optimizer.apply_gradients(zip(grads, trainable_vars))

                if (e_i % self.log_period == 0):
                    self._log_protocol(L, e_i + (e * num_images // self.batch_size))

                if (e_i % self.save_period == 0):
                    self._save_protocol(e_i + (e * num_images // self.batch_size))
            print("Epoch complete. Backing up log.")
        print("Training finished.")

    def _get_A_base_outputs(self):
        img = tf.convert_to_tensor(ImageUtils.get_image(self.style_path), tf.float32)
        img = tf.expand_dims(img, 0)
        img = self.vgg.preprocess(img)
        return self.vgg.model(img)

    def _forward_pass(self, X):
        X_proc = self.transform.preprocess(X)
        assert (X_proc.get_shape()[0] == self.batch_size)
        # T for transformed now that transformed
        T = self.transform.model(X_proc)
        T = self.transform.unprocess(T)

        T_proc = self.vgg.preprocess(T)
        T_vgg_out = self.vgg.model(T_proc)
        T_content_o = T_vgg_out[:self.vgg.partition_idx]
        T_style_o = T_vgg_out[self.vgg.partition_idx:]

        preproc_imgs = self.vgg.preprocess(X)
        X_vgg_out = self.vgg.model(preproc_imgs)
        X_content_o = X_vgg_out[:self.vgg.partition_idx]

        L = self._get_loss(T_content_o, T_style_o, X_content_o, T)
        return L

    # TODO: Might not need this
    @tf.function
    def _get_loss(self, T_content, T_style, X_content, T_img):

        content_loss = self._get_content_loss(T_content, X_content)
        style_loss = self._get_style_loss(T_style)
        tv_loss = self._get_total_variation_loss(T_img)

        L_style = style_loss * self.style_weight
        L_content = content_loss * self.content_weight
        L_tv = tv_loss * self.tv_weight

        total_loss = L_style + L_content + L_tv

        return Loss(total_loss=total_loss,
                    style_loss=L_style,
                    content_loss=L_content,
                    tv_loss=L_tv)

    def _gram_matrix(self, input_tensor, shape=None):
        # Modified from Google Demo
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        if shape:
            num_locations = shape[1] * shape[2] * shape[3]
        else:
            input_shape = tf.shape(input_tensor)
            num_locations = input_shape[1] * input_shape[2] * input_shape[3]
        num_locations = tf.cast(num_locations, tf.float32)
        return result / num_locations

    def _get_content_loss(self, T_content, X_content):
        content_loss = 0
        assert (len(T_content) == len(X_content))
        for i, T in enumerate(T_content):
            weight = self.c_layer_weights[i]
            B, H, W, CH = T.get_shape()  # first return value is batch size (must be one)
            HW = H * W  # product of width and height
            inst_cont_loss = weight * 2 * tf.nn.l2_loss(T - X_content[i]) / (B * HW * CH)
            content_loss += inst_cont_loss
        return content_loss

    def _get_style_loss(self, T_style):
        style_loss = 0
        assert (len(T_style) == len(self.A_style_grams))
        for i, T in enumerate(T_style):
            weight = self.s_layer_weights[i]
            B, H, W, CH = T.get_shape()  # first return value is batch size (must be one)
            HW = H * W  # product of width and height

            G = self._gram_matrix(T, (B, HW, CH))  # style feature of x
            A = self.A_style_grams[i]  # style feature of a
            style_loss += weight * 2 * tf.nn.l2_loss(G - A) / (B * (CH ** 2))
        return style_loss


    def _get_total_variation_loss(self, img):
        return tf.reduce_sum(tf.image.total_variation(img))

    def _log_protocol(self, L, iteration):
        tf.print("iteration: %d, total_loss: %f, style_loss: %f, content_loss: %f, tv_loss: %f" \
                 % (iteration, L.total_loss, L.style_loss, L.content_loss, L.tv_loss))
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iteration, L.total_loss, L.style_loss, L.content_loss, L.tv_loss])
        # TODO: find better backup for logs
        if (iteration % (self.log_period * 10) == 0):
            back_up_log_dest = os.path.join(self.log_dest_path, "logs_%s.csv" % str(iteration))
            copy2(self.log_file, back_up_log_dest)

    def _save_protocol(self, iteration):
        dest_path = os.path.join(self.save_dest_path, str(iteration))
        tf.keras.models.save_model(model=self.transform.model, filepath=self.saved_model_path)
        copy_tree(self.saved_model_path, dest_path)

