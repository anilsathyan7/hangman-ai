import runpy
import sys


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--model",
        "canine",
        "--secret",
        "QUIZ SHOW",
        "--pattern",
        "Q___ S___",
        *sys.argv[1:],
    ]
    runpy.run_module("test_multitask", run_name="__main__")
