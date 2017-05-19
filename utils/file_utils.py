import glob
import tensorflow as tf
from tensorflow.python.lib.io import file_io


def file_list(paths, image_format='.jpg'):
    # Get names of training cases - sample files
    print("Files will be loaded from: \"" + paths + '*' + image_format + "\"")
    filenames = sorted(file_io.get_matching_files(paths + '*' + image_format))

    if not filenames:
        raise IOError("Files not found!")

    return filenames


def load_rgb(queue, random_crop=True, randomize=False):
    key, file = tf.WholeFileReader().read(queue)
    uint8_image = tf.image.decode_jpeg(file, channels=3, name='decoded_uint8_image')
    if random_crop:
        uint8_image = tf.random_crop(uint8_image, (224, 224, 3))
    if randomize:
        uint8_image = tf.image.random_flip_left_right(uint8_image)
        uint8_image = tf.image.random_flip_up_down(uint8_image, seed=None)
    float_image = tf.div(tf.cast(uint8_image, tf.float32), 255)

    return float_image


def pipeline(paths, batch_size=1, epochs=None, min_after_dequeue=100):
    if paths is None:
        pass

    queue = tf.train.string_input_producer(paths, num_epochs=epochs, shuffle=False)
    files = load_rgb(queue, randomize=False)
    capacity = min_after_dequeue + 3 * batch_size
    batch = tf.train.shuffle_batch([files], batch_size=batch_size, capacity=capacity,
                                   min_after_dequeue=min_after_dequeue)

    return batch
