from argparse import ArgumentParser
from src.train_model import TrainTweetTypeModel, TrainTweetRoleModel


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--classifier', type=str, help='Type of classifier to train, type or role')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train for', default=50)
    parser.add_argument('--l2_values', nargs='+', type=float,
                        help='List of l2_values to train a model for')
    args = parser.parse_args()

    if args.classifier == 'type':
        trainer = TrainTweetTypeModel()
    elif args.classifier == 'role':
        trainer = TrainTweetRoleModel()
    else:
        raise ValueError("classifier must be 'role' or 'type'")

    for l2_value in args.l2_values:
        trainer.output_model(l2_value, args.epochs)
