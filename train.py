import tensorflow as tf
import argparse
import math
import os
import matplotlib.pyplot as plt

from tqdm import tqdm

from networks.loss import depth_loss
from networks.model import MVAAutoEncoder

import dataset
import utils


# Argument Parser
parser = argparse.ArgumentParser(description='Monocular Depth Estimation based on AutoEncoder with Attention Model')

parser.add_argument('--bs',
                    type=int,
                    default=4,
                    help='Batch size.')
parser.add_argument('--lr',
                    type=float,
                    default=0.0001,
                    help='Learning rate.')
parser.add_argument('--epochs',
                    type=int,
                    default=20,
                    help='Number of epochs.')
parser.add_argument('--logs',
                    type=str,
                    default='./logs/',
                    help='Directory for tensorflow summary.')
parser.add_argument('--ckpt',
                    type=str,
                    default='./checkpoint/',
                    help='Directory for model checkpoint.')
parser.add_argument('--gpu',
                    type=str,
                    default='0',
                    help='GPU Device.')

args = parser.parse_args()


def prepare_summary(color_path, depth_path):
    color_list = os.listdir(color_path)
    depth_list = os.listdir(depth_path)

    summary_color_list = []
    summary_depth_list = []
    for color in color_list:
        color = tf.io.read_file(color_path + color)
        decoded_color = tf.image.decode_jpeg(color)
        preprocess_color = tf.image.convert_image_dtype(decoded_color, dtype=tf.float32)
        preprocess_color = tf.expand_dims(preprocess_color, axis=0)
        summary_color_list.append(preprocess_color)

    for depth in depth_list:
        depth = tf.io.read_file(depth_path + depth)
        decoded_depth = tf.image.decode_jpeg(depth)
        decoded_depth = tf.image.resize(decoded_depth, [240, 320])
        preprocess_depth = tf.image.convert_image_dtype(decoded_depth, dtype=tf.float32)
        preprocess_depth = tf.expand_dims(preprocess_depth, axis=0)
        summary_depth_list.append(preprocess_depth)

    return summary_color_list, summary_depth_list


def main():
    # For tensorboard summary image
    summary_color_list, summary_depth_list = prepare_summary('./summary_image/color/', './summary_image/depth/')

    # Generate dataset
    train_nyu_length, train_nyu, test_nyu_length, test_nyu = dataset.loader(args.bs, '../data/nyu2_train.csv', '../data/nyu2_test.csv')

    # Build Model
    mva = MVAAutoEncoder()
    model = mva.build_model()
    
    print('\nModel built...')
    print('    -> AutoEncoder with Attention model (with pre-trained DenseNet).')

    # Print model summary
    utils.print_model(model)

    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(args.lr)

    # Tensorflow summaries
    writer = tf.summary.create_file_writer(args.logs)

    @tf.function
    def train_step(rgb_batch, depth_batch):
        with tf.GradientTape() as tape:
            predictions = model(rgb_batch)
            loss = depth_loss(y_true=depth_batch, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def valid_step(rgb_batch, depth_batch):
        predictions = model(rgb_batch)
        loss = depth_loss(y_true=depth_batch, y_pred=predictions)
        return loss

    def summaries(epoch, loss, valid_loss, rgbs, gts, preds):
        with writer.as_default():
            tf.summary.scalar('Train Loss',
                              loss,
                              step=epoch,
                              description='train loss')
            tf.summary.scalar('Valid Loss',
                              valid_loss,
                              step=epoch,
                              description='valid loss')
            tf.summary.image('RGB images',
                             rgbs,
                             step=epoch,
                             description='RGB image')
            tf.summary.image('Ground-truths',
                             utils.visualize(gts),
                             step=epoch,
                             description='Ground truth depth')
            tf.summary.image('Predicted Depths',
                             utils.visualize(preds),
                             step=epoch,
                             description='Predicted depth')

    # Start training
    print('\nTraining start...')
    for epoch in tqdm(range(args.epochs), desc='Monocular Depth'):
        step = 0
        for rgbs, depths in train_nyu:
            step += 1
            train_loss = train_step(rgbs, depths)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}".format(epoch,
                                                                   args.epochs,
                                                                   step,
                                                                   math.ceil(train_nyu_length / args.bs),
                                                                   train_loss))
        for rgbs, depths in test_nyu:
            valid_loss = valid_step(rgbs, depths)

        print("Epoch: {}/{}, train_loss: {:.5f}, valid_loss: {:.5f}".format(epoch,
                                                                            args.epochs,
                                                                            train_loss,
                                                                            valid_loss))

        summaries(epoch, 
                  train_loss, 
                  valid_loss,
                  tf.concat([summary_color_list[0], summary_color_list[1], summary_color_list[2]], 0),
                  tf.concat([summary_depth_list[0], summary_depth_list[1], summary_depth_list[2]], 0), 
                  tf.concat([model.predict(summary_color_list[0]), model.predict(summary_color_list[1]), model.predict(summary_color_list[2])], 0)
        )
        writer.flush()

        if epoch % 5 == 0:
            model.save_weights(filepath=args.ckpt + 'epoch{}_{}.h5'.format(epoch, valid_loss), save_format='h5')

    # Save model weights
    print('\n Training done...')
    print('    -> Model saved (./checkpoint/model.h5)')
    model.save_weights(filepath=args.ckpt + 'model.h5', save_format='h5')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main()