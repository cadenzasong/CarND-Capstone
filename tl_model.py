"""
Traffic Light Detection Network Implementation
submitted by team Prithvi Nuthanakalva

This module generates, trains and tests a traffic light detection network
derived from the VGG-16 network.
"""
###############################################################################
import os.path
import tensorflow as tf
import warnings
from distutils.version import LooseVersion
import yaml
import re
import random
import numpy as np
import scipy.misc
import shutil
import zipfile
import time
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

#==============================================================================
class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

#==============================================================================
def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

#==============================================================================
def gen_batch_function(data_dir, data_desc, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """

        data_list = yaml.load(open(os.path.join(data_dir, data_desc), 'r'))
        image_path_gt_list = []
        for data in data_list:
            path = os.path.join(data_dir, data['path'])
            boxes = data['boxes']
            gt = 3 # No light=3, green=2, yellow=1, red light=0
            for box in boxes:
                if box['label'][0:2] == 'Red':
                    if gt != 3: continue # Skip images with different traffic lights
                    gt = 0
                if box['label'][0:5] == 'Yellow':
                    if gt != 3: continue # Skip images with different traffic lights
                    gt = 1
                if box['label'][0:4] == 'Green':
                    if gt != 3: continue # Skip images with different traffic lights
                    gt = 2
            image_path_gt_list.append([path, gt])

        random.shuffle(image_path_gt_list)
        # Next line is needed to generate only complete batches
        batch_len = (len(image_path_gt_list) // batch_size) * batch_size
        for batch_i in range(0, batch_len, batch_size):
            images = []
            gts = [] # No light, green, yellow, red light
            for image_path_gt in image_path_gt_list[batch_i:batch_i+batch_size]:
                image = scipy.misc.imresize(scipy.misc.imread(image_path_gt[0]), image_shape)

                images.append(image)
                gts.append(image_path_gt[1])

            yield np.array(images), np.array(gts)
    return get_batches_fn

#==============================================================================
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer7_out)
    """
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path);

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    graph = tf.get_default_graph();
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name);
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name);
    layer_7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name);

    return image_input, keep_prob, layer_7_out

#==============================================================================
def layers(vgg_layer7_out, num_classes):
    """
    Create the layers for am image classification network.
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Flatten VGG layer 7
    print(vgg_layer7_out.get_shape())
    vgg_flat = tf.reshape(vgg_layer7_out, [-1, 5 * 18 * 4096])
    print(vgg_flat.get_shape())

    # Dense Layer
    dense1 = tf.layers.dense(vgg_flat, units=512, activation=tf.nn.relu)
    print(dense1.get_shape())
    dropout1 = tf.layers.dropout(dense1, rate=0.4)
    print(dropout1.get_shape())
    dense2 = tf.layers.dense(dropout1, units=64, activation=tf.nn.relu)
    print(dense2.get_shape())
    dropout2 = tf.layers.dropout(dense2, rate=0.4)
    print(dropout2.get_shape())

    # Logits Layer
    logits = tf.layers.dense(dropout2, units=num_classes)
    print(logits.get_shape())

    return logits

#==============================================================================
def optimize(logits, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    one_hot = tf.one_hot(correct_label, num_classes)
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return train_op, accuracy_op

#==============================================================================
def train_test_nn(sess, epochs, batch_size,
                  get_train_batch_fn, train_op, get_test_batch_fn, test_op,
                  input_image, correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_train_batch_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param get_test_batch_fn: Function to get batches of validation data.  Call using get_batches_fn(batch_size)
    :param test_op: TF Operation to validate the neural network
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())

    print("Training...")
    print()
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))

        for image, label in get_train_batch_fn(batch_size):
            sess.run(train_op, feed_dict={input_image: image, correct_label: label, learning_rate: 0.0005, keep_prob: 0.6})
            print('-', end='', flush=True)
        print()

        total_accuracy = 0
        batch_num = 0

        for image, label in get_test_batch_fn(batch_size):
            accuracy = sess.run(test_op, feed_dict={input_image: image, correct_label: label, keep_prob: 0.6})
            total_accuracy += accuracy
            batch_num += 1
            print('=', end='', flush=True)
        print()

        validation_accuracy = total_accuracy / batch_num
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

#==============================================================================
def run():
    num_classes = 4
    image_shape = (160, 576)
    data_dir = './data'
    batch_size = 6
    epoch_num = 2

    # Download pretrained vgg model
    maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_train_batch_fn = gen_batch_function(os.path.join(data_dir, 'train'), 'train.yaml', image_shape)
        get_test_batch_fn = gen_batch_function(os.path.join(data_dir, 'test'), 'test.yaml', image_shape)

        # Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.int32, (None))
        learning_rate = tf.placeholder(tf.float32)

        image_input, keep_prob, layer_7_out = load_vgg(sess, vgg_path)
        logits = layers(layer_7_out, num_classes)
        train_op, accuracy_op = optimize(logits, correct_label, learning_rate, num_classes)

        # Train+test NN using the train_test_nn function
        train_test_nn(sess, epoch_num, batch_size,
                      get_train_batch_fn, train_op, get_test_batch_fn, accuracy_op,
                      image_input, correct_label, keep_prob, learning_rate)


#==============================================================================
if __name__ == '__main__':
    run()
