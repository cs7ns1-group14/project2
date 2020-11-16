#!/usr/bin/env python3
# convertf.py - convert TensorFlow model to TensorFlow Lite

# Authors:
#   Basil Contovounesios <contovob@tcd.ie>
#   Salil Kulkarni <sakulkar@tcd.ie>

import tensorflow as tf
import tensorflow.keras as keras

import argparse

__version__ = '0.0.1'


def get_args():
    opt = argparse.ArgumentParser(description='convert TF model to TFLite')
    opt.add_argument('-V', '--version', action='version',
                     version=f'%(prog)s {__version__}')
    opt.add_argument('model', help='input TF model name')
    return opt.parse_args()


def load_model(name):
    with open(f'{name}.json') as f:
        model = f.read()
    model = keras.models.model_from_json(model)
    model.load_weights(f'{name}.h5')
    return model


def main():
    args = get_args()
    cvt = tf.lite.TFLiteConverter.from_keras_model(load_model(args.model))
    tflite = cvt.convert()
    with open(f'{args.model}.tflite', 'wb') as f:
        f.write(tflite)


if __name__ == '__main__':
    main()
