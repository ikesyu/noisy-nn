from multidimensional_storage_plot import *


def multidim_fit_freq(dimbin, plot_shuffle):
    dimpatt = number_to_binary_list(dimbin)
    dims = range(1, 3)
    losses = [{True: [], False: []} for d in dims]
    for dim in dims:
        structure = Structure([2**(6//dim)]*dim)
        nfuncs = []
        for i in range(1, 101):

            ns, cs = function_shape(i, dimpatt)
            # print(f"# {dimbin=} {dimpatt=} {ns=} {cs=}")
            if ns is None:
                continue
            nfuncs.append(i)
            for shuffle in [True, False]:
                function = PlainSineFunctions(construct_shape=cs,
                                              present_shape=reshape_dims(
                                                  ns, dim),
                                              epochs=10000,
                                              shuffle=shuffle, shuffle_learning=True)
                loss = retrieve(structure, function)["losses"]
                losses[dim-1][shuffle].append(np.max(loss))
        print(f"# {nfuncs=}")

    if plot_shuffle:
        shuffles = [False, True]
    else:
        shuffles = [False]

    for s in shuffles:
        for dim in dims:
            os = "shuffled" if s else "ordered"
            if s:
                pf = plt.semilogy
            else:
                pf = plt.plot
            pf(nfuncs,  losses[dim-1][s],
               ".-", label=f"V={dim} {os}")

    plt.xlabel("Number of functions")
    plt.xlim([0, 100])
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    fs = [l for l, v in zip(["Phase", "Frequency", "Amplitude"], dimpatt) if v]
    # plt.title(f"Function space {fs}")
    ps = "s" if plot_shuffle else "n"
    plt.savefig(f"../fig/multidim_fit_freq{dimbin}{ps}.pdf")
    plt.close()


for dimbin in range(1, 4):
    for plot_shuffle in [False, True]:
        multidim_fit_freq(dimbin, plot_shuffle)
