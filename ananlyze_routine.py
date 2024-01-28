from os import makedirs
from ananlyze_routine_imp import *
from stiv_compute_routine import (
    root,
)
from values import ananlyze_result_dir


def main():
    TestMode = 0
    if TestMode == 0:
        ananlyze_result_wrong()
    else:
        print("Unknown method")


if __name__ == "__main__":
    main()
