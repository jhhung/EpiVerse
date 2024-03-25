from HiConformer.utils.Logger import get_logger
import os


def create_output_dirs(output_path: str) -> None:
    for subdir in ["models", "checkpoints", "contactmaps"]:
        os.makedirs(os.path.join(output_path, subdir), exist_ok=True)