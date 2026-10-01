import runpy
import sys


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--model",
        "slimbert",
        "--secret",
        "GENUINE EXPERIENCE",
        "--pattern",
        "_EN__NE E__E__EN_E",
        *sys.argv[1:],
    ]
    runpy.run_module("test_multitask", run_name="__main__")
