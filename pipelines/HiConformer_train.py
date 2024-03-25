import subprocess
import argparse

def main():
# Initialize the ArgumentParser
    parser = argparse.ArgumentParser(description='Script to run HiConformer train')
    parser.add_argument('--config', type=str, default="../HiConformer/Experiments/Training_config.yaml", required=True, help="Path to the training configuration file.")
    parser.add_argument('--device', type=str, default=0, help="CUDA visible devices. Default is 0.", required=True)
    parser.add_argument('--version', type=str, default="V001", help="User defined version name. Default is V001.", required=True)
    parser.add_argument('--tissue', type=str, default="IMR90_MboI", help="Training Tissue name.", required=True)
    parser.add_argument('--outdir', type=str, default="../training_results", help="Output directory for training results. Default is ../training_results.", required=True)

    args = parser.parse_args()

    # Construct the command to run train.py
    command = [
        'python', '../HiConformer/train.py',
        '--config', args.config,
        '--device', args.device,
        '--version', args.version,
        '--tissue', args.tissue,
        '--outdir', args.outdir
    ]

    # Run the command
    subprocess.run(command)

if __name__ == "__main__":
    main()