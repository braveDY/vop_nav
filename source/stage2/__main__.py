import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(description='Stage two: collect data, train VOP-Net, compare predictions')
    parser.add_argument('command', choices=('collect', 'train', 'compare'))
    args = parser.parse_args(sys.argv[1:2])
    module = importlib.import_module(f'stage2.{args.command}.run')
    sys.argv = [f'{sys.argv[0]} {args.command}', *sys.argv[2:]]
    module.main()


if __name__ == '__main__':
    main()
