import argparse
import logging
from pathlib import Path

from cal_ratio_trainer.config import load_config


def do_train(args):
    # Get the config loaded.
    c = load_config(args.config)

    # Next, look at the arguments and see if anything should be changed.
    if args.epochs is not None:
        c.epochs = args.epochs
    if args.num_splits is not None:
        c.num_splits = args.num_splits

    from cal_ratio_trainer.training.runner_utils import training_runner_util

    training_runner_util(c)


def main():
    # Create the master command line parser
    parser = argparse.ArgumentParser(
        description="Training and Utilities for the CalRatio RNN training"
    )
    subparsers = parser.add_subparsers(help="sub-command help")
    parser.set_defaults(func=lambda _: parser.print_help())

    # Add a top level verbose flag that will count the number of
    # v's on the command line to turn on levels of verbosity.
    parser.add_argument("-v", "--verbose", action="count", default=0)

    # Sub-command train to actually run the training, using a config file to load.
    parser_train = subparsers.add_parser("train", help="Train the CalRatio RNN model")
    parser_train.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to the config file to use for training",
    )
    # Add all the training configuration options
    parser_train.add_argument("--epochs", type=int, help="Number of epochs to train")
    parser_train.add_argument("--num_splits", type=int, help="Number of mini-batches")
    parser_train.set_defaults(func=do_train)

    # Parse the command line arguments
    args = parser.parse_args()

    # Turn on verbosity by setting the log level to be "info" or "debug"
    # in python's `logging` module. Shut down matplotlib, it makes
    # way too much noise.
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    else:
        log_level = logging.WARNING
    logging.basicConfig(level=log_level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    # Call the function to execute the command.
    args.func(args)


if __name__ == "__main__":
    main()
