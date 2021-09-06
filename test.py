import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import argparse
import os

from PIL import Image

from networks.model import MVAAutoEncoder


parser = argparse.ArgumentParser(description='Monocular Depth Estimation based on AutoEncoder with Attention Model')

parser.add_argument('--model_weights',
                    type=str,
                    default='./checkpoint/model.h5',
                    help='Model weights path.')
parser.add_argument('--gpu',
                    type=str,
                    default='0',
                    help='GPU Device.')

args = parser.parse_args()


def load_image(path):
    image = np.asarray(Image.open(path))
    return image


def preprocessing(image):
    # Normalize image [0, 1]
    image = image / 255.

    height, width, _ = image.shape
    
    # Convert ndarray to tf tensor
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)

    # If height != 480 and width != 640, Resize image
    # Network was trained by (480, 640) fixed image size for Multi-View Attention
    if height != 480 and width != 640:
        image = tf.image.resize(image, [480, 640])
    
    return image


def visualize_depth(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = depth * 255.
    depth = depth.astype(np.uint8)
    return depth


def main():
    # image_dir can change to your own directory.
    image_dir = './examples'
    image_list = os.listdir('./examples')

    # Build model and Load model weights
    print('Load MVA Network...')
    net = MVAAutoEncoder()
    model = net.build_model()
    model.load_weights('./checkpoint/model.h5')

    for image_name in image_list:
        path = os.path.join(image_dir, image_name)

        # Load image data
        image = load_image(path)
        ori_height, ori_width, _ = image.shape

        # Preprocess image
        image = preprocessing(image)

        print('{}, ({}, {}) >> Depth Estimation...'.format(image_name, ori_height, ori_width))
        # Inference depth image
        pred_depth = model(image)

        # Resize the depth image to original image size
        pred_depth = tf.image.resize(pred_depth, [ori_height, ori_width])
        # Depth normalize [10, 1000]
        pred_depth = tf.clip_by_value((1000 / pred_depth), 10, 1000)
        pred_depth = tf.reshape(pred_depth, [ori_height, ori_width])
        pred_depth = pred_depth.numpy()
        
        visual_depth = visualize_depth(pred_depth)
        Image.fromarray(visual_depth).save('./results/{}.png'.format(image_name.split('.')[0] + '_depth'))


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main()