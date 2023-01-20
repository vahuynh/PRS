import numpy as np
import time
from multiprocessing import Pool
from rase import rase


def rase_network(dataLS, dataVS, estimator, estimator_kwargs, input_indices=None, normalize_data=True,
                 niterations=10, nmodels=100, B=500, D=None, C0=0.1, random_seed=100, nthreads=1):

    '''

    Gene network inference with RASE.

    :param dataLS: gene expression, learning dataset (n_samples_LS, ngenes)
    :param dataVS: gene expression, validation dataset (n_samples_VS, ngenes).
    :param estimator: scikit-learn learner class (type of model).
    :param estimator_kwargs: dictionary containing the hyper-parameters of the estimator.
    :param input_indices: indices of the candidate regulators among the genes. If None, all the genes are candidate regulators.
    :param normalize_data: boolean indicating if input data must be normalized.
    :param niterations: number of iterations of the RaSE algorithm.
    :param nmodels: number of base models in the ensemble.
    :param B: number of subspace candidates generated for each base model.
    :param D: maximal subspace size when generating random subspaces. If None, D is set to min(sqrt(n_samples_LS), nfeatures).
    :param C0: positive constant used to set the minimum feature selection probability.
    :param random_seed: random seed
    :param nthreads: number of threads used for parallel computing.

    :return: a tuple(VIM, subspace_sizes)
    - VIM: array of shape (nreg, ngenes), where the element [i,j] is the importance of the i-th candidate regulator for the j-th target gene. nreg is the number of candidate regulators.
    - subspace_sizes: array of shape(nmodels, ngenes), where the j-th column contains the subspace sizes for the j-th trained ensemble.
    '''

    time_start = time.time()

    ngenes = dataLS.shape[1]

    # Get the indices of the candidate regulators
    if input_indices is None:
        input_indices = np.arange(ngenes)

    nregulators = len(input_indices)

    # Do RaSE for each target gene
    VIM = np.zeros((nregulators, ngenes))
    subspace_sizes = np.zeros((nmodels, ngenes))

    if nthreads > 1:
        print('running jobs on %d threads' % nthreads)

        input_data = list()
        for i in range(ngenes):
            input_data.append([dataLS, dataVS, i, input_indices, estimator, estimator_kwargs, normalize_data, niterations, nmodels, B, D, C0, random_seed])

        with Pool(nthreads) as pool:
            alloutput = pool.starmap(rase_network_single, input_data)

        for (i, (vi, subspace_sizes_i)) in enumerate(alloutput):
            VIM[:, i] = vi
            subspace_sizes[:, i] = subspace_sizes_i

    else:
        print('running single threaded jobs')
        for i in range(ngenes):
            print('Gene %d/%d...' % (i + 1, ngenes))

            (VIM[:, i], subspace_sizes[:, i]) = rase_network_single(dataLS, dataVS, i, input_indices,
                                                                    estimator, estimator_kwargs,
                                                                    normalize_data, niterations,
                                                                    nmodels, B, D, C0, random_seed)

    time_end = time.time()
    print("Elapsed time: %.2f seconds" % (time_end - time_start))

    return VIM, subspace_sizes


def rase_network_single(dataLS, dataVS, output_idx, input_idx, estimator, estimator_kwargs, normalize_data, niterations, nmodels, B, D, C0, random_seed):

    ngenes = dataLS.shape[1]

    # Expression of target gene
    yLS = dataLS[:, output_idx]
    yVS = dataVS[:, output_idx]

    # Remove target gene from candidate regulators
    input_idx = list(input_idx[:])
    input_idx_noOutput = input_idx.copy()
    if output_idx in input_idx_noOutput:
        input_idx_noOutput.remove(output_idx)

    XLS = dataLS[:, input_idx_noOutput]
    XVS = dataVS[:, input_idx_noOutput]

    (vi_tmp, subspace_sizes, yTSpred) = rase(XLS, yLS, XVS, yVS,
                                             estimator=estimator, estimator_kwargs=estimator_kwargs,
                                             isClassification=False, nmodels=nmodels, B=B, D=D,
                                             niterations=niterations, C0=C0,
                                             normalize_data=normalize_data, random_seed=random_seed)


    vi = np.zeros(ngenes)
    vi[input_idx_noOutput] = vi_tmp

    return vi[input_idx], subspace_sizes
