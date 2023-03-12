import keras
import argparse
# from tools.train_net import train


def argument_parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "mode", default="train", help="determine to train, test or inference", type=str,
    )
    args_ = parser.parse_args()
    return args_


if __name__ == '__main__':
    args = argument_parse()
    if args.mode == 'train':
        print("="*30, "start training", "="*30)
        train()
    elif args.mode == 'test':
        pass
    elif args.mode == 'inference':
        pass
    else:
        raise Exception(f"{args.mode} is not exists")
