import argparse
import gzip

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--test_size', type=int, required=False, default=10000)

    args = parser.parse_args()

    with gzip.open(args.input_path, 'r') as inp_file:
        with gzip.open('test_' + args.out_path, 'w') as out_file:
            for _ in tqdm(range(args.test_size)):
                line = inp_file.readline()
                out_file.write(line)
        print('Test size', args.test_size)
        test_size = 0
        with gzip.open('train_' + args.out_path, 'w') as out_file:
            for line in tqdm(inp_file):
                out_file.write(line)
                test_size += 1
    print('Train size', test_size)


if __name__ == '__main__':
    main()
