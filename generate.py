#!/usr/bin/env python3
# generate.py - generate CAPTCHA images for model training

# Based on a modified version by Ciarán Mc Goldrick.

# Authors:
#   Basil Contovounesios <contovob@tcd.ie>
#   Salil Kulkarni <sakulkar@tcd.ie>

from captcha.image import ImageCaptcha

import argparse
import base64
import os
import secrets

__version__ = '0.0.1'


def get_args():
    opt = argparse.ArgumentParser(
        description='Generate CAPTCHAs for training.')
    opt.add_argument('-V', '--version', action='version',
                     version=f'%(prog)s {__version__}')
    opt.add_argument('-l', '--length', type=int, default=6,
                     help='CAPTCHA character length (default: %(default)s)')
    opt.add_argument('-H', '--height', type=int, default=64,
                     help='CAPTCHA height in pixels (default: %(default)s)')
    opt.add_argument('-W', '--width', type=int, default=128,
                     help='CAPTCHA width in pixels (default: %(default)s)')

    req = opt.add_argument_group('required arguments')
    req.add_argument('-S', '--symbol-file', metavar='FILE', required=True,
                     help='file with CAPTCHA symbols to use')
    req.add_argument('-c', '--count', type=int, required=True,
                     help='number of CAPTCHAs to generate')
    req.add_argument('-o', '--output', metavar='DIR', required=True,
                     help='directory to store generated CAPTCHAs')

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


def get_captcha_name(args):
    string = ''.join(secrets.choice(args.symbols) for _ in range(args.length))
    encoded = base64.urlsafe_b64encode(string.encode()).decode()
    path = os.path.join(args.output, f'{encoded}.png')
    version = 1
    while os.path.exists(path):
        path = os.path.join(args.output, f'{encoded}_{version}.png')
        version += 1
    return string, path


def main():
    args = get_args()
    print('Using symbol set:', args.symbols)
    image = ImageCaptcha(width=args.width, height=args.height)
    os.makedirs(args.output, exist_ok=True)
    abs_dir = os.path.abspath(args.output)
    progress = reporter(f'Generating CAPTCHAs in {abs_dir}', args.count)
    for i in range(args.count):
        image.write(*get_captcha_name(args))
        progress(i + 1)


if __name__ == '__main__':
    main()
