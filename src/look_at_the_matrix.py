import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from entropy_and_mi import plot
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature selection using QUBO and Mutual Information."
                                                 " The user can visualize the Qubo matrix, which is composed "
                                                 "by summing the Redundancy and Importance matrices, "
                                                 "by inserting the name of the file corresponding to Q.")

    parser.add_argument("dir", type=str, help="Directory to inspect files.")
    parser.add_argument("file", type=str, nargs="?", help="File name of the Qubo matrix to visualize.")
    parser.add_argument('--list', help='List all the files in the directory', action='store_true')

    args = parser.parse_args()

    if args.list:
        if not os.path.isdir(args.dir):
            print(f"Error: Directory '{args.dir}' does not exist.")
            exit(1)

        print("\n".join(os.listdir(args.dir)))
        exit(0)

    if not args.file:
        print("Error: No file specified. Provide a file name as the second argument.")
        exit(1)

    file_path = os.path.join(args.dir, args.file)

    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        exit(1)

    try:
        q_matrix = np.loadtxt(file_path, delimiter=",")
    except Exception as e:
        print(f"Error loading the file: {e}")
        exit(1)

    plot(q_matrix)
    plt.show()