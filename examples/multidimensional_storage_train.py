import argparse
import torch
import numpy as np
from multidimensional_storage import Structure,  train_net, set_cuda
from multidimensional_storage_functions import SineFunctions, SquineFunctions, TrineFunctions
import time


def underbar_list(li):
    return '_'.join([f"{n}" for n in li])


def get_filename(structure, functions):
    ns = underbar_list(structure.noise_structure)
    fs = underbar_list(functions.shape)

    if functions.shuffle_learning:
        shuffle = "3" if functions.shuffle else "2"
    else:
        shuffle = "1" if functions.shuffle else "0"

    label = f"n{ns}_a{structure.activation_ratio}" + \
        f"_f{fs}_s{shuffle}_e{functions.epochs}"
    if hasattr(functions, "function_type"):
        cs = underbar_list(functions.construct_shape)
        label += f"_t{functions.function_type}"
        label += f"_c{cs}"
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
        '--construct_structure',
        nargs='+',             # Expect one or more arguments
        type=int,              # Convert each argument to int
        help='number of functions in parameter space',
        default=None
    )

    parser.add_argument(
        '--shuffle', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        '--shuffle_learning', action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--function_type", default="sine")

    parser.add_argument('--activation_ratio', type=float,
                        default=0.5, help='Percentange of neurons active')

    parser.add_argument('--epochs', type=int,
                        default=10000, help='Epochs per function')

    parser.add_argument(
        "--cuda", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    if args.construct_structure is None:
        args.construct_structure = args.function_structure

    set_cuda(args.cuda)

    structure = Structure(args.noise_structure,
                          args.activation_ratio)

    print(f"{args.function_type=}")
    if args.function_type == "sine":
        functions = SineFunctions(args.function_structure,
                                  shuffle=args.shuffle, epochs=args.epochs,
                                  shuffle_learning=args.shuffle_learning
                                  )
    elif args.function_type == "squine":
        functions = SquineFunctions(construct_shape=args.construct_structure,
                                    present_shape=args.function_structure,
                                    shuffle=args.shuffle, epochs=args.epochs,
                                    shuffle_learning=args.shuffle_learning
                                    )
    elif args.function_type == "trine":
        functions = TrineFunctions(construct_shape=args.construct_structure,
                                   present_shape=args.function_structure,
                                   shuffle=args.shuffle, epochs=args.epochs,
                                   shuffle_learning=args.shuffle_learning
                                   )

    else:
        raise Exception(f"{args.function_type}:no such function type")
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
