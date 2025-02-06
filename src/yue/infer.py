import os
import sys

from common import create_args


def check_exit(status: int):
    if status != 0:
        sys.exit(status)


if __name__ == "__main__":
    _, parser = create_args()  # make --help work

    dirname = os.path.dirname(os.path.abspath(__file__))
    print("Starting stage 1...")
    check_exit(os.system(f'python {os.path.join(dirname, "infer_stage1.py")} {" ".join(sys.argv[1:])}'))
    print("Starting stage 2...")
    check_exit(os.system(f'python {os.path.join(dirname, "infer_stage2.py")} {" ".join(sys.argv[1:])}'))
    print("Starting postprocessing...")
    check_exit(os.system(f'python {os.path.join(dirname, "infer_postprocess.py")} {" ".join(sys.argv[1:])}'))
