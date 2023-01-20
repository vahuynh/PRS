import numpy as np
from numpy.random import seed, permutation, choice
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.dummy import DummyRegressor
import utils

def PRS_network(dataLS, dataVS, estimator, estimator_kwargs, input_indices=None, normalize_data=True,
                alphas_init=None, lambda_reg=0, nmodels=100, batch_size=None, train_size=None,
                nepochs=3000, learning_rate=0.001, rho1=0.9, rho2=0.999, variance_reduction=True, random_seed=100):

    '''

    Gene network inference with PRS.

    :param dataLS: gene expression, learning dataset (n_samples_LS, ngenes).
    :param dataVS: gene expression, validation dataset (n_samples_VS, ngenes).
    :param estimator: scikit-learn estimator class (type of base model).
    :param estimator_kwargs: dictionary containing the hyper-parameters of the estimator.
    :param input_indices: indices of the candidate regulators among the genes. If None, all the genes are candidate regulators.
    :param normalize_data: boolean indicating if data must be normalized.
    :param alphas_init: array of shape (nreg, ngenes), where nreg is the number of candidate regulators and ngenes is the number of target genes, containing initial values of the parameters alphas. If one scalar, then all alphas at initialized at that scalar. If None, alphas are initialized to 5/nmodels.
    :param lambda_reg: value of the regularisation coefficient for structured sparsity constraint (hyper-parameter lambda in the paper).
    :param nmodels: number of base models in each PRS ensemble.
    :param batch_size: number of samples in each mini-batch. If None, the mini-batch size is set to 10% of the training set size.
    :param train_size: number of samples used to train each base model. If None, train_size = original training set size - batch_size.
    :param nepochs: number of epochs of the training algorithm.
    :param learning_rate: learning rate of the Adam algorithm.
    :param rho1: hyper-parameter rho1 of the Adam algorithm.
    :param rho2: hyper-parameter rho1 of the Adam algorithm.
    :param variance_reduction: boolean indicating whether or not to apply the variance reduction technique with baseline
    :param random_seed: integer specifying the random seed.
    :return: a tuple (alphas, objective_values_LS, objective_values_VS, error_VS, train_indices)
    - alphas: array of shape (nreg, ngenes). Element [i, j] is the probability of selecting the i-th candidate regulator in the model predicting the j-th gene. nreg is the number of candidate regulators.
    - objective_values_LS: array of shape (nepochs+1,)  containing the value of the objective function at each epoch on the learning set.
    - objective_values_VS: array of shape (nepochs+1,)  containing the value of the objective function at each epoch on the validation set.
    - error_VS: mean squared error on the validation set
    - train_indices: indices of the epochs where new models were learned.
    '''


    (nsamples_LS, nfeat) = dataLS.shape

    if normalize_data:
        # Normalize data
        scaler = StandardScaler()
        dataLS = scaler.fit_transform(dataLS)
        dataVS = scaler.transform(dataVS)

    model_package = dict()
    model_package['model'] = estimator
    model_package['model_kwargs'] = estimator_kwargs
    model_package['constant_estimator'] = DummyRegressor(strategy='mean')
    model_package['ensemble_aggregate'] = utils.ensemble_average
    model_package['loss_gradient'] = utils.grad_loss_regression
    model_package['variance_reduction'] = variance_reduction

    if alphas_init is None:
        alphas_init = 5/nmodels

    if batch_size is None:
        batch_size = int(np.round(dataLS.shape[0] * 0.1))

    if input_indices is None:
        input_indices = np.arange(nfeat)

    (alphas, objective_values_LS, objective_values_VS, train_indices) = optimise_probabilities(dataLS, dataVS, model_package, alphas_init, lambda_reg, nmodels, batch_size, train_size, nepochs, learning_rate, rho1, rho2, input_indices, random_seed)

    # Prediction of validation set
    # Sample feature subsets
    ninputs = len(input_indices)
    Z = np.zeros((nmodels, ninputs, nfeat), dtype=bool)
    for j in range(nfeat):
        Z[:, :, j] = utils.sample_features(alphas[:, j], nmodels)

    estimators = utils.network_fit(dataLS, input_indices, Z, model_package)
    data_pred_VS = utils.network_predict(estimators, input_indices, Z, dataVS)
    error_VS = ((dataVS - data_pred_VS) ** 2).mean()

    return alphas, objective_values_LS, objective_values_VS, error_VS, train_indices


def optimise_probabilities(dataLS, dataVS, model_package, alphas_init, lambda_reg, nmodels, batch_size, train_size, nepochs, learning_rate, rho1, rho2, input_indices, random_seed, eps=1e-8):

    '''
    PRS training (one run).
    '''

    # Set the random seed
    seed(random_seed)

    nsamples, nfeat = dataLS.shape
    ninputs = len(input_indices)

    nsplits = int(np.round(nsamples / batch_size))

    beta_min = 1 / nmodels

    objective_values_LS = np.zeros(nepochs + 1)
    objective_values_VS = np.zeros(nepochs + 1)
    train_indices = [0]

    betas = np.zeros((ninputs, nfeat)) + alphas_init
    # A feature can not be used to predict itself
    for j in range(ninputs):
        betas[j, input_indices[j]] = 0
    alphas = betas.copy()
    betas_best = betas.copy()

    p_ratio = np.ones((nmodels, nfeat))

    idxperm = permutation(nsamples)
    dataLS = dataLS[idxperm, :]
    mb_data = list()
    data_pred_mb_expected = list()
    kf = KFold(n_splits=nsplits, shuffle=False)
    lossLS = np.zeros(nfeat)
    lossVS = np.zeros(nfeat)

    for k, (train_index, test_index) in enumerate(kf.split(dataLS)):

        # Sample feature subsets
        Z = np.zeros((nmodels, ninputs, nfeat), dtype=bool)
        for j in range(nfeat):
            Z[:, :, j] = utils.sample_features(alphas[:, j], nmodels)

        # Features that have been selected for at least one base model (check that for each ensemble)
        Z_selected = np.sum(Z, axis=0)
        feat_idx_to_optimise = Z_selected > 0

        # Learn new models and predict samples in mini-batch
        if train_size is not None:
            train_index = choice(train_index, size=train_size, replace=False)

        data_train, data_mb = dataLS[train_index, :], dataLS[test_index, :]

        estimators = utils.network_fit(data_train, input_indices, Z, model_package)
        data_pred_mb = utils.network_predict(estimators, input_indices, Z, data_mb)
        data_pred_VS = utils.network_predict(estimators, input_indices, Z, dataVS)

        mb_data.append((data_mb, data_pred_mb, data_pred_VS, Z, feat_idx_to_optimise))

        # Initial value of the objective function, when alphas are not optimised yet
        data_pred_mb_expected.append(np.mean(data_pred_mb, axis=0))
        data_pred_VS_expected = np.mean(data_pred_VS, axis=0)
        lossLS = lossLS + np.mean((data_mb - data_pred_mb_expected[k]) ** 2, axis=0)
        lossVS = lossVS + np.mean((dataVS - data_pred_VS_expected) ** 2, axis=0)

    objective_values_LS[0] = np.mean(lossLS / nsplits) + lambda_reg * np.sum(np.sqrt(np.sum(betas ** 2, axis=1)))
    objective_values_VS[0] = np.mean(lossVS / nsplits) + lambda_reg * np.sum(np.sqrt(np.sum(betas ** 2, axis=1)))
    objective_value_VS_min = objective_values_VS[0]

    n = 0  # epoch number
    t = 0  # step number

    first_moment = np.zeros_like(betas)
    second_moment = np.zeros_like(betas)

    while n < nepochs:

        n += 1
        lossLS = np.zeros(nfeat)
        lossVS = np.zeros(nfeat)

        for k, (data_mb, data_pred_mb, data_pred_VS, Z, feat_idx_to_optimise) in enumerate(mb_data):

            t += 1

            grad = gradient(data_mb, data_pred_mb_expected[k], data_pred_mb, Z, betas, p_ratio, lambda_reg, feat_idx_to_optimise, model_package)

            # Estimation of the projected gradient
            betas_new = betas - eps * grad
            betas_new[betas_new < 0] = 0
            betas_new[betas_new > 1] = 1
            grad_proj = (betas - betas_new) / eps

            # Adam optimization step
            first_moment = rho1 * first_moment + (1 - rho1) * grad_proj
            second_moment = rho2 * second_moment + (1 - rho2) * grad_proj ** 2
            first_moment_hat = first_moment / (1 - rho1 ** t)
            second_moment_hat = second_moment / (1 - rho2 ** t)
            betas = betas - learning_rate * first_moment_hat / (np.sqrt(second_moment_hat) + eps)

            # Final projection (to correct for numerical errors)
            betas[betas < 0] = 0
            betas[betas > 1] = 1

            # Avoid the critical situation where all the betas drop to zeros
            # (Usually due to the fact that the learning rate is too high)
            for j in range(nfeat):
                if sum(betas[:, j]) == 0:
                    print('!!!Reinializating alphas for gene %d!!!' % (j + 1))
                    betas[:, j] = beta_min
                    betas[j, input_indices[j]] = 0

            p_ratio = compute_p_ratio(alphas, betas, Z)
            data_pred_mb_expected[k] = expected_output(data_pred_mb, p_ratio)
            data_pred_VS_expected = expected_output(data_pred_VS, p_ratio)
            lossLS = lossLS + np.mean((data_mb - data_pred_mb_expected[k]) ** 2, axis=0)
            lossVS = lossVS + np.mean((dataVS - data_pred_VS_expected) ** 2, axis=0)

        objective_values_LS[n] = np.mean(lossLS / nsplits) + lambda_reg * np.sum(np.sqrt(np.sum(betas ** 2, axis=1)))
        objective_values_VS[n] = np.mean(lossVS / nsplits) + lambda_reg * np.sum(np.sqrt(np.sum(betas ** 2, axis=1)))

        if objective_values_VS[n] < objective_value_VS_min:
            objective_value_VS_min = objective_values_VS[n]
            betas_best = betas.copy()

        nmodels_effective = np.zeros(nfeat)
        denom = np.sum(p_ratio ** 2, axis=0)
        if sum(denom == 0) == 0:
            nmodels_effective = np.sum(p_ratio, axis=0) ** 2 / denom

        if sum(nmodels_effective < 0.9 * nmodels) > 0:

            alphas = betas.copy()
            p_ratio = np.ones((nmodels, nfeat))

            # Train new models

            train_indices.append(n)

            idxperm = permutation(nsamples)
            dataLS = dataLS[idxperm, :]
            mb_data = list()
            kf = KFold(n_splits=nsplits, shuffle=False)

            for k, (train_index, test_index) in enumerate(kf.split(dataLS)):

                # Sample feature subsets
                Z = np.zeros((nmodels, ninputs, nfeat), dtype=bool)
                for j in range(nfeat):
                    Z[:, :, j] = utils.sample_features(alphas[:, j], nmodels)

                # Features that have been selected for at least one base model (check that for each ensemble)
                Z_selected = np.sum(Z, axis=0)
                feat_idx_to_optimise = Z_selected > 0

                # Learn new models and predict samples in mini-batch
                if train_size is not None:
                    train_index = choice(train_index, size=train_size, replace=False)

                data_train, data_mb = dataLS[train_index, :], dataLS[test_index, :]

                estimators = utils.network_fit(data_train, input_indices, Z, model_package)
                data_pred_mb = utils.network_predict(estimators, input_indices, Z, data_mb)
                data_pred_VS = utils.network_predict(estimators, input_indices, Z, dataVS)

                mb_data.append((data_mb, data_pred_mb, data_pred_VS, Z, feat_idx_to_optimise))
                data_pred_mb_expected[k] = expected_output(data_pred_mb, p_ratio)

    return betas_best, objective_values_LS, objective_values_VS, train_indices


def compute_p_ratio(alphas, betas, Z, eps=1e-6):
    """
    :param alphas:
    :param betas:
    :param Z:
    :return:
    """

    nmodels = Z.shape[0]

    alphas = np.tile(alphas, (nmodels, 1, 1))
    betas = np.tile(betas, (nmodels, 1, 1))

    alphas[~Z] = 1 - alphas[~Z]
    betas[~Z] = 1 - betas[~Z]

    alphas = np.clip(alphas, a_min=eps, a_max=None)

    p_ratio = np.prod(betas/alphas, axis=1)
    p_ratio = np.clip(p_ratio, 0, 1e50)  # To avoid overflows

    return p_ratio


def expected_output(data_pred, p_ratio):

    """
    Computes the expectations of the outputs
    """

    (nmodels, nsamples, nfeat) = data_pred.shape

    data_pred_expected = np.transpose(np.tile(p_ratio, (nsamples, 1, 1)), axes=[1, 0, 2]) * data_pred
    data_pred_expected = np.mean(data_pred_expected, axis=0)

    return data_pred_expected


def gradient(data, data_pred_expected, data_pred, Z, betas, p_ratio, lambda_reg, feat_idx_to_optimise, model_package, eps=1e-6):

    """
    Compute the gradient of the objective function w.r.t. alphas.
    """

    (nmodels, nsamples, nfeat) = data_pred.shape
    ninputs = betas.shape[0]
    grad = np.zeros((ninputs, nfeat))

    for j in range(nfeat):

        feat_idx_to_optimise_j = np.where(feat_idx_to_optimise[:, j])[0]
        nfeat_to_optimise = len(feat_idx_to_optimise_j)

        # Score function
        h = np.tile(betas[feat_idx_to_optimise_j, j], (nmodels, 1))
        h[~Z[:, feat_idx_to_optimise_j, j]] = h[~Z[:, feat_idx_to_optimise_j, j]] - 1

        # Positive values too close to zero are set to eps
        h[Z[:, feat_idx_to_optimise_j, j]] = np.clip(h[Z[:, feat_idx_to_optimise_j, j]], a_min=eps, a_max=None)
        # Negative values too close to zero are set to -eps
        h[~Z[:, feat_idx_to_optimise_j, j]] = np.clip(h[~Z[:, feat_idx_to_optimise_j, j]], a_min=None, a_max=-eps)

        h = 1 / h

        data_pred_j = np.transpose(np.tile(data_pred[:, :, j], (nfeat_to_optimise, 1, 1)), axes=(1, 2, 0))

        if model_package['variance_reduction']:

            h_square = h ** 2
            h_var = np.mean(np.reshape(p_ratio[:, j], (-1, 1)) * h_square, axis=0)
            h_var = np.clip(h_var, a_min=eps, a_max=None)

            h_square = np.transpose(np.tile(h_square, (nsamples, 1, 1)), axes=(1, 0, 2))

            b = np.mean(np.reshape(p_ratio[:, j], (-1, 1, 1)) * data_pred_j * h_square, axis=0) / h_var.reshape(1, -1)

        else:

            b = 0

        h = np.transpose(np.tile(h, (nsamples, 1, 1)), axes=(1, 0, 2))
        grad_output = np.mean(np.reshape(p_ratio[:, j], (-1, 1, 1)) * (data_pred_j - b) * h, axis=0)

        # For the gradient of the square root, we set 0/0 = 0
        reg_grad_num = betas[feat_idx_to_optimise_j, j]
        reg_grad_denom = np.sqrt(np.sum(betas[feat_idx_to_optimise_j, :]**2, axis=1))
        grad[feat_idx_to_optimise_j, j] = model_package['loss_gradient'](data[:, j], data_pred_expected[:, j], grad_output) / nfeat + lambda_reg * np.divide(reg_grad_num, reg_grad_denom, out=np.zeros_like(reg_grad_num), where=reg_grad_denom != 0)

    return grad
