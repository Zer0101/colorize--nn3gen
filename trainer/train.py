import os
import sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from utils.model_utils import init_model
from utils.image_utils import concat_images


def train(configs):
    print(configs)
    print(configs.save_path)
    if configs.save_model is not None and configs.save_model:
        if not os.path.exists(configs.save_path):
            os.makedirs(configs.save_path)
        print("Model will be saved to \"" + configs.save_path + "\" ")

    if configs.save_images is not None and configs.save_images:
        if not os.path.exists(configs.output_path):
            os.makedirs(configs.output_path)
        print("Images will be saved to \"" + configs.output_path + "\" ")

    if not os.path.exists(configs.log_dir):
        os.makedirs(configs.log_dir)
    print("Logs will be saved to \"" + configs.log_dir + "\" ")

    batch_size, epochs, global_step, learning_rate, phase_train, uv, color_image_rgb, color_image_yuv, \
    grayscale_image, grayscale_image_rgb, grayscale_image_yuv, grayscale, weights, graph_def, \
    imported_graph_defs, graph, tensors, model, last_layer, last_layer_yuv, last_layer_rgb, loss = init_model(configs)

    # Run the optimizer
    if phase_train is not None:
        optimizer = tf.train.GradientDescentOptimizer(0.0001)
        opt = optimizer.minimize(loss, global_step=global_step, gate_gradients=optimizer.GATE_NONE)

    # Create a session for running operations in the Graph.
    session = tf.InteractiveSession()

    # Create the graph, etc.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Saver
    saver = tf.train.Saver()
    # Initialize the variables.

    session.run(init_op)

    # Merge summary
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(configs.log_dir, session.graph)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)

    # If continue is set we reinit from checkpoint
    if configs.continue_train:
        # Continue to train model
        if configs.model_fullpath is None:
            print("Cannot find the model. Please specify path to model")
            exit()

        # Load the checkpoint
        ckpt = tf.train.get_checkpoint_state(configs.model_fullpath)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            print("Load model finished!")
        else:
            print("Failed to restore model")
            exit()

    # Start a number of threads, passing the coordinator to each of them.
    with coord.stop_on_exception():
        while not coord.should_stop():
            print('Running training...')
            # Run training steps
            training_opt = session.run(opt, feed_dict={phase_train: True, uv: 1})
            training_opt = session.run(opt, feed_dict={phase_train: True, uv: 2})

            step = session.run(global_step)
            if step % 1 == 0:
                last_layer_, last_layer_rgb_, color_image_rgb_, grayscale_image_rgb_, cost, merged_ = session.run(
                    [last_layer, last_layer_rgb, color_image_rgb, grayscale_image_rgb, loss, merged],
                    feed_dict={phase_train: False, uv: 3})

                print("Running step: %d" % step)
                print("Cost: %f" % np.mean(cost))

                if configs.save_images is not None and configs.save_images:
                    if configs.output_save_step is not None \
                            and step % configs.output_save_step == 0:
                        print("Saving images...")
                        images_format = configs['images']['output']['format']
                        summary_image = concat_images(grayscale_image_rgb_[0], last_layer_rgb_[0])
                        summary_image = concat_images(summary_image, color_image_rgb_[0])

                        step_prefix = str(step)
                        image_name_prefix = configs.output_path + step_prefix

                        plt.imsave(image_name_prefix + "_summary" + images_format, summary_image)
                        plt.imsave(image_name_prefix + "_grayscale" + images_format,
                                   grayscale_image_rgb_[0])
                        plt.imsave(image_name_prefix + "_color" + images_format, color_image_rgb_[0])
                        plt.imsave(image_name_prefix + "_colorized" + images_format,
                                   last_layer_rgb_[0])

                        print("Saved image at run: %d" % step)
                        sys.stdout.flush()

                sys.stdout.flush()
                writer.add_summary(merged_, step)
                writer.flush()
            if configs.save_model is not None and configs.save_model:
                if configs.save_model_step is not None and step % configs.save_model_step == 0:
                    print("Saving model...")
                    saver.save(session, configs.save_path + 'model.ckpt')
                    print("Model saved in file: %s" % configs.save_path + 'model.ckpt')
                    sys.stdout.flush()

    # Wait for threads to finish.
    coord.join(threads)
    session.close()