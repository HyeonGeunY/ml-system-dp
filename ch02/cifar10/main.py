import argparse
import os

import mlflow

def main():
    parser = argparse.ArgumentParser(
        description="Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--commit_hash",
        type=str,
        default="000000",
        help="code commit hash",
    )
    
