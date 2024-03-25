import argparse
import subprocess

def init_parser():
    parser = argparse.ArgumentParser(description='MIRNet for Hi-C data enhancement.')
    parser.add_argument('--image_size', type=int, default=200, help='Size of the input images.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs for training.')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--train_data_dir', type=str, required=True, help='Directory containing training data.')
    parser.add_argument('--val_data_dir', type=str, required=True, help='Directory containing validation data.')
    parser.add_argument('--result_dir', type=str, default='../training_results', help='Directory to save the results and model checkpoints.')
    parser.add_argument("--device", type=str, default="0", help="GPU device to use.")
    return parser.parse_args()

if __name__ == "__main__":
    args = init_parser()
    process = subprocess.run(["python", "../MIRNet/MIRNet_train.py",
                    "--image_size", f"{args.image_size}",
                    "--batch_size", f"{args.batch_size}",
                    "--epochs", f"{args.epochs}",
                    "--learning_rate", f"{args.learning_rate}",
                    "--train_data_dir", f"{args.train_data_dir}",
                    "--val_data_dir", f"{args.val_data_dir}",
                    "--result_dir", f"{args.result_dir}",
                    "--device", f"{args.device}"])
    print("MIRNet training completed.")