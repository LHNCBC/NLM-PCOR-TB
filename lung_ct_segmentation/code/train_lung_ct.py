import subprocess
import argparse
import sys

"""
This training file is used to run train lung segmentation model on CT volumes.
 ou can run 5 fold cross validation models by providing fold number
(in 0,1,2,3,4) in the argument .You can also train a model by
providing all of the available images to training by assigning the value 'all' in
the cv_fold_number argument.
"""


def main():

    parser = argparse.ArgumentParser("Preprocessing steps using nnunet")
    parser.add_argument(
        "--task_number",
        required=True,
        type=int,
        help="Task number(XXX), where the folder name is stored in \
                            nnUNet_raw_data_base/nnUNet_raw_data/TaskXXX_MYTASK format",
    )
    parser.add_argument(
        "--cv_fold_number",
        required=True,
        type=int,
        help="Cross validation fold number to train the model. \
                            Select any number in (0,1,2,3,4).You can also provide \
                            the value 'all",
    )
    args = parser.parse_args()

    subprocess.call(
        [
            "nnUNet_train ",
            "2d",
            "nnUNetTrainerV2",
            "Task" + args.task_number + "_MYTASK ",
            args.cv_fold_number,
            "--npz",
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
