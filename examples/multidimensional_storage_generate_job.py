from multidimensional_storage import Structure, SineFunctions


def spaced_list(l):
    return " ".join([f"{s}" for s in l])


def bool_param(label, val):
    p = "" if val else "no-"
    return f"--{p}{label}"


def get_job_string(structure, functions, cuda):
    args = [
        "python multidimensional_storage_train.py",
        f"--noise_structure {spaced_list(structure.noise_structure)}",
        f"--function_structure {spaced_list(functions.shape)}",
        bool_param("shuffle", functions.shuffle),
        bool_param("cuda", cuda),
        f"--epochs {functions.epochs}",
        f"--activation_ratio {structure.activation_ratio}"
    ]

    return " ".join(args)


def unidim_comparison():
    structure = Structure([100], activation_ratio=0.5)
    cuda = True
    for i in range(1, 101):
        for shuffle in [False, True]:
            functions = SineFunctions([i], shuffle=shuffle, epochs=10000)
            print(get_job_string(structure, functions, cuda))


# unidim_comparison()
