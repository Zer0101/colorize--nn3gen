from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
from utils.image_utils import from_rgb_to_yuv, from_yuv_to_rgb, from_grayscale_to_rgb, from_rgb_to_grayscale
from utils.file_utils import pipeline, file_list
from models.colorize import Colorize


def init_model(config):
    batch_size = tf.constant(config.batch_size, name='batch_size')
    # Initialize number of epochs - very important and sensitive value
    epochs = tf.constant(config.epochs, name='global_epochs')
    # Create global step value. It will change automatically with in tensorflow context
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Activate learning rate with decay - NN with batch normalization MUST have decaying leaning rate
    learning_rate = tf.train.exponential_decay(config.learning_rate, global_step,
                                               config.learning_decay,
                                               config.learning_decay_step, staircase=True)
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    uv = tf.placeholder(tf.uint8, name='uv')

    paths = []
    try:
        paths = file_list(config.inputs, config.input_format)
    except IOError as e:
        print("There is no files to read")
        exit()

    color_image_rgb = pipeline(paths, config.batch_size, config.epochs)
    color_image_yuv = from_rgb_to_yuv(color_image_rgb)
    grayscale_image = from_rgb_to_grayscale(color_image_rgb)
    grayscale_image_rgb = from_grayscale_to_rgb(grayscale_image)
    grayscale_image_yuv = from_rgb_to_yuv(grayscale_image_rgb)
    grayscale = tf.concat([grayscale_image, grayscale_image, grayscale_image], 3, 'grayscale_image_tensor')

    # Initializing tensor with weights
    weights = {
        # 1x1 convolution, 512 inputs, 256 outputs
        'wc1': tf.Variable(tf.truncated_normal([1, 1, 512, 256], stddev=0.01)),
        # 3x3 convolution, 512 inputs, 128 outputs
        'wc2': tf.Variable(tf.truncated_normal([3, 3, 256, 128], stddev=0.01)),
        # 3x3 convolution, 256 inputs, 64 outputs
        'wc3': tf.Variable(tf.truncated_normal([3, 3, 128, 64], stddev=0.01)),
        # 3x3 convolution, 128 inputs, 3 outputs
        'wc4': tf.Variable(tf.truncated_normal([3, 3, 64, 3], stddev=0.01)),
        # 3x3 convolution, 6 inputs, 3 outputs
        'wc5': tf.Variable(tf.truncated_normal([3, 3, 3, 3], stddev=0.01)),
        # 3x3 convolution, 3 inputs, 2 outputs
        'wc6': tf.Variable(tf.truncated_normal([3, 3, 3, 2], stddev=0.01)),
    }

    # # Create a session for running operations in the Graph.
    # session = tf.InteractiveSession()

    graph_def = tf.GraphDef()
    # We load trained classification model to use it intermediary convolution layers
    # This pass will is only for first training
    # Every training continuation will use model generated from VGG-16
    try:
        with open(config.vgg, mode='rb') as file:
            print('Loaded VGG-16 model')
            file_content = file.read()
            graph_def.ParseFromString(file_content)
            file.close()
    except IOError as e:
        print('Cannot find VGG-16 model. Training is stopped')
        exit()
    finally:
        sys.stdout.flush()

    imported_graph_defs = tf.import_graph_def(graph_def, input_map={"images": grayscale})
    graph = tf.get_default_graph()

    with tf.variable_scope('vgg'):
        conv1_2 = graph.get_tensor_by_name("import/conv1_2/Relu:0")
        conv2_2 = graph.get_tensor_by_name("import/conv2_2/Relu:0")
        conv3_3 = graph.get_tensor_by_name("import/conv3_3/Relu:0")
        conv4_3 = graph.get_tensor_by_name("import/conv4_3/Relu:0")
    tensors = {
        "conv1_2": conv1_2,
        "conv2_2": conv2_2,
        "conv3_3": conv3_3,
        "conv4_3": conv4_3,
        "grayscale": grayscale,
        "weights": weights
    }
    model = Colorize(tensors=tensors, phase=phase_train)

    # Get weights from model
    last_layer = model.get_last_layer()
    # Transform them to RGB
    last_layer_yuv = tf.concat(values=[tf.split(axis=3, num_or_size_splits=3,
                                                value=grayscale_image_yuv)[0], last_layer], axis=3)
    # transform yuv back to RGB
    last_layer_rgb = from_yuv_to_rgb(last_layer_yuv)

    # Calculate the loss
    loss = tf.square(tf.subtract(last_layer, tf.concat(
        [tf.split(axis=3, num_or_size_splits=3, value=color_image_yuv)[1],
         tf.split(axis=3, num_or_size_splits=3, value=color_image_yuv)[2]], 3)))
    if uv == 1:
        loss = tf.split(axis=3, num_or_size_splits=2, value=loss)[0]
    elif uv == 2:
        loss = tf.split(axis=3, num_or_size_splits=2, value=loss)[1]
    else:
        loss = (tf.split(axis=3, num_or_size_splits=2, value=loss)[0] + tf.split(axis=3, num_or_size_splits=2,
                                                                                 value=loss)[1]) / 2
    # # Run the optimizer
    # if phase_train is not None:
    #     optimizer = tf.train.GradientDescentOptimizer(0.0001)
    #     opt = optimizer.minimize(loss, global_step=global_step, gate_gradients=optimizer.GATE_NONE)

    # Summaries
    tf.summary.histogram(name="weights1", values=weights["wc1"])
    tf.summary.histogram(name="weights2", values=weights["wc2"])
    tf.summary.histogram(name="weights3", values=weights["wc3"])
    tf.summary.histogram(name="weights4", values=weights["wc4"])
    tf.summary.histogram(name="weights5", values=weights["wc5"])
    tf.summary.histogram(name="weights6", values=weights["wc6"])
    tf.summary.histogram(name="instant_loss", values=tf.reduce_mean(loss))
    tf.summary.image(name="colorimage", tensor=color_image_rgb, max_outputs=1)
    tf.summary.image(name="pred_rgb", tensor=last_layer_rgb, max_outputs=1)
    tf.summary.image(name="grayscale", tensor=grayscale_image_rgb, max_outputs=1)

    return batch_size, epochs, global_step, learning_rate, phase_train, uv, color_image_rgb, color_image_yuv, \
           grayscale_image, grayscale_image_rgb, grayscale_image_yuv, grayscale, weights, graph_def, \
           imported_graph_defs, graph, tensors, model, last_layer, last_layer_yuv, last_layer_rgb, loss
