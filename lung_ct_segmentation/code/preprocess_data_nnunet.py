import subprocess
import argparse
import sys

"""
As a preprocessing steps nnunet extracts a dataset fingerprint (a set of dataset-specific
properties such as image sizes, voxel spacings, intensity information etc) which in
turn is used to create 3 network configurations.(2D,3D,3DCascade) UNets
"""


def main():

    parser = argparse.ArgumentParser("Preprocessing steps using nnunet")
    parser.add_argument(
        "task_number",
        required=True,
        type=str,
        help="Task number(XXX), where the folder name is stored in \
                            nnUNet_raw_data_base/nnUNet_raw_data/TaskXXX_MYTASK format",
    )
    args = parser.parse_args()

    subprocess.call(
        [
            "nnUNet_plan_and_preprocess",
            "-t",
            args.task_number,
            " --verify_dataset_integrity",
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
