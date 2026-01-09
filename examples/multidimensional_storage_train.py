import argparse
import torch
import numpy as np
from multidimensional_storage import Structure, SineFunctions, train_net, set_cuda
import time


def get_filename(structure, functions):
    ns = '_'.join([f"{n}" for n in structure.noise_structure])
    fs = '_'.join([f"{n}" for n in functions.shape])
    shuffle = "1" if functions.shuffle else "0"
    label = f"n{ns}_a{structure.activation_ratio}_f{
        fs}_s{shuffle}_e{functions.epochs}"
    fname = f"../data/multidim_{label}.pt"
    return fname


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process a list of integers.')
    parser.add_argument(
        '--noise_structure',
        nargs='+',             # Expect one or more arguments
        type=int,              # Convert each argument to int
        help='structure',
        default=[100]
    )

    parser.add_argument(
        '--function_structure',
        nargs='+',             # Expect one or more arguments
        type=int,              # Convert each argument to int
        help='number of functions',
        default=[10]
    )

    parser.add_argument(
        '--shuffle', action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument('--activation_ratio', type=float,
                        default=0.5, help='Percentange of neurons active')

    parser.add_argument('--epochs', type=int,
                        default=10000, help='Epochs per function')

    parser.add_argument(
        "--cuda", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    set_cuda(args.cuda)

    structure = Structure(args.noise_structure,
                          args.activation_ratio)
    functions = SineFunctions(args.function_structure,
                              shuffle=args.shuffle, epochs=args.epochs)

    fname = get_filename(structure, functions)
    print(f"Training {fname}")

    start = time.perf_counter()
    model, losses = train_net(structure, functions)
    elapsed = time.perf_counter() - start
    print(f"Elapsed: {elapsed:.6f} s")
    print(f"max loss {np.max(losses)}")

    torch.save({"model": model, "losses":
                losses}, fname
               )
