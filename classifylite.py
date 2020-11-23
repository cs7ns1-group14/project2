#!/usr/bin/env python3
# classifylite.py - classify CAPTCHA images using TensorFlow Lite

# Based on a modified version by Ciarán Mc Goldrick.

# Authors:
#   Basil Contovounesios <contovob@tcd.ie>
#   Salil Kulkarni <sakulkar@tcd.ie>

from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite

import argparse
import os

__version__ = '0.0.1'


def get_args():
    opt = argparse.ArgumentParser(
        description='Classify CAPTCHAs using a TensorFlow Lite model.')
    opt.add_argument('-V', '--version', action='version',
                     version=f'%(prog)s {__version__}')

    req = opt.add_argument_group('required arguments')
    req.add_argument('-m', '--model', metavar='FILE', required=True,
                     help='TFLite model to use for classification')
    req.add_argument('-d', '--directory', required=True,
                     help='directory with CAPTCHAs to classify')
    req.add_argument('-o', '--output', metavar='FILE', required=True,
                     help='output file')
    req.add_argument('-S', '--symbol-file', metavar='FILE', required=True,
                     help='file with CAPTCHA symbols to use')

    args = opt.parse_args()
    with open(args.symbol_file) as f:
        args.symbols = f.readline().strip('\n')
    return args


def reporter(msg, total):
    def update(i, prefix='\b\b\b\b'):
        pct = int(i * 100 / total)
        end = '' if i < total else '\n'
        print(f'{prefix}{pct:3}%', end=end, flush=True)
    update(0, f'{msg}...')
    return update


def decode(symbols, predictions):
    indices = np.argmax(predictions, axis=2)[:, 0]
    return ''.join(symbols[i] for i in indices).replace(' ', '')


def main():
    args = get_args()
    print('Using symbol set:', args.symbols)
    interpreter = tflite.Interpreter(args.model)
    interpreter.allocate_tensors()
    index = interpreter.get_input_details()[0]['index']
    out_details = interpreter.get_output_details()

    abs_dir = os.path.abspath(args.directory)
    captchas = sorted(os.listdir(args.directory))
    progress = reporter(f'Classifying CAPTCHAs in {abs_dir}', len(captchas))

    with open(args.output, 'w') as out_file:
        for i, captcha in enumerate(captchas):
            img = Image.open(os.path.join(args.directory, captcha))
            img = np.array(img, np.float32) / 255.0
            img = img.reshape((-1, *img.shape))
            interpreter.set_tensor(index, img)
            interpreter.invoke()
            predictions = [interpreter.get_tensor(out['index'])
                           for out in out_details]
            prediction = decode(args.symbols, predictions)
            out_file.write(f'{captcha},{prediction}\n')
            progress(i + 1)


if __name__ == '__main__':
    main()
