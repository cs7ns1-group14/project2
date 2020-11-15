#!/usr/bin/env python3

import argparse
import csv

__version__ = '0.0.1'


def get_args():
    opt = argparse.ArgumentParser(
        description='Merge two CSV files into a third using the first column.')
    opt.add_argument('-V', '--version', action='version',
                     version=f'%(prog)s {__version__}')
    opt.add_argument('in1', help='first input CSV file')
    opt.add_argument('in2', help='second input CSV file')
    opt.add_argument('out', help='merged output CSV file')
    return opt.parse_args()


def read_csv_dict(path):
    with open(path, newline='') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONE)
        return {row[0]: row[1] for row in reader}


def write_csv_dict(path, dictionary):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE, quotechar='',
                            lineterminator='\n')
        for key in sorted(dictionary):
            writer.writerow((key, dictionary[key]))


def main():
    args = get_args()
    in1 = read_csv_dict(args.in1)
    in1.update(read_csv_dict(args.in2))
    write_csv_dict(args.out, in1)


if __name__ == '__main__':
    main()
