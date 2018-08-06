import argparse
import os
import numpy as np

from model import denoise
import tensorflow as tf

parser = argparse.ArgumentParser(description='')

parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=10, help='# images in batch')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')

args = parser.parse_args()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    low_res_img_count = 49
    with tf.Session() as sess:
        model = denoise(sess)

        if args.phase == 'train':
            model.train_model()
        else:
            print "Testing facility not ready"

if __name__ == '__main__':
    tf.app.run()
