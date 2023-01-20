import numpy as np
from numpy.random import seed, permutation, choice
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import mean_squared_error
import utils

def PRS(XLS, yLS, XVS, yVS, estimator, estimator_kwargs, isClassification, normalize_data=True, XTS=None,
        alphas_init=None, lambda_reg=0, nmodels=100, batch_size=None, train_size=None,
        nepochs=3000, learning_rate=0.001, rho1=0.9, rho2=0.999,
        variance_reduction=True, random_seed=100):

    '''

    PRS model training

    :param XLS: array of shape (n_samples_LS, n_features) containing the inputs of the learning set.
    :param yLS: array of shape (n_samples_LS,) containing the outputs of the learning set (class labels in classification, real numbers in regression).
    :param XVS: array of shape (n_samples_VS, n_features) containing the inputs of the validation set.
    :param yVS: array of shape (n_samples_VS,) containing the outputs of the validation set (class labels in classification, real numbers in regression).
    :param estimator: scikit-learn learner class.
    :param estimator_kwargs: dictionary containing the hyper-parameters of the estimator.
    :param isClassification: boolean indicating if this is a classification problem or not (otherwise it is a regression problem)
    :param normalize_data: boolean indicating if input data must be normalized.
    :param XTS: array of shape (n_samples_TS, nfeatures) containing the inputs of the test set.
    :param alphas_init: vector containing initial values of the parameters alphas. If one scalar, then all alphas are initialized at that scalar. If None, alphas are initialized to 5/nmodels.
    :param lambda_reg: value of regularisation coefficient (hyper-parameter lambda in the paper).
    :param nmodels: number of models in the PRS ensemble.
    :param batch_size: number of samples in each mini-batch. If None, the mini-batch size is set to 10% of the training set size.
    :param train_size: number of samples used to train each base model. If None, train_size = original training set size - batch_size.
    :param nepochs: number of epochs of the training algorithm.
    :param learning_rate: learning rate of the Adam algorithm.
    :param rho1: hyper-parameter rho1 of the Adam algorithm.
    :param rho2: hyper-parameter rho1 of the Adam algorithm.
    :param variance_reduction: boolean indicating whether or not to apply the variance reduction technique with baseline
    :param random_seed: integer specifying the random seed.
    :return: a tuple (alphas, objective_values_LS, objective_values_VS, train_indices, yVSpred, yTSpred)
    - alphas: array of shape (nfeatures,), containing the trained alphas values.
    - objective_values_LS: array of shape (nepochs+1,) containing the value of the objective function at each epoch on the learning set.
    - objective_values_VS: array of shape (nepochs+1,) containing the value of the objective function at each epoch on the validation set.
    - train_indices: indices of the epochs where new models were learned.
    - yVSpred: array of shape (n_samples_VS,) containing the predictions of the trained ensemble model on the validation set.
    - yTSpred: array of shape (n_samples_TS,) containing the predictions of the trained ensemble model on the test set.
      '''

    if normalize_data:
        # Normalize data
        X_scaler = StandardScaler()
        XLS = X_scaler.fit_transform(XLS)
        XVS = X_scaler.transform(XVS)
        if XTS is not None:
            XTS = X_scaler.transform(XTS)

    model_package = dict()
    model_package['model'] = estimator
    model_package['model_kwargs'] = estimator_kwargs
    model_package['variance_reduction'] = variance_reduction



    if isClassification:

        # Classification
        nclasses = len(np.unique(yLS))

        # Transform labels into integers
        le = LabelEncoder()
        yLS = le.fit_transform(yLS)
        yVS = le.transform(yVS)

        model_package['constant_estimator'] = DummyClassifier(strategy='most_frequent')
        model_package['KFold'] = StratifiedKFold
        model_package['ensemble_prediction'] = utils.ensemble_prediction_classification
        model_package['loss_metric'] = utils.cross_entropy
        model_package['ensemble_aggregate'] = utils.class_probability
        model_package['stratify'] = yLS
        model_package['loss_gradient'] = utils.grad_loss_classification
        model_package['expected_output_gradient'] = utils.grad_expected_output_classification
        model_package['nclasses'] = nclasses
    else:
        # Regression
        model_package['constant_estimator'] = DummyRegressor(strategy='mean')
        model_package['KFold'] = KFold
        model_package['ensemble_prediction'] = utils.ensemble_prediction_regression
        model_package['loss_metric'] = mean_squared_error
        model_package['ensemble_aggregate'] = utils.ensemble_average
        model_package['stratify'] = None
        model_package['loss_gradient'] = utils.grad_loss_regression
        model_package['expected_output_gradient'] = utils.grad_expected_output_regression
        model_package['nclasses'] = None

    # Initial alphas
    if alphas_init is None:
        alphas_init = 5/nmodels

    if batch_size is None:
        batch_size = int(np.round(XLS.shape[0]*0.1))

    # PRS training
    alphas, objective_values_LS, objective_values_VS, train_indices = optimise_probabilities(XLS, yLS, XVS, yVS, model_package, alphas_init, lambda_reg, nmodels, batch_size, train_size, nepochs, learning_rate, rho1, rho2, random_seed)

    # Predict on validation set and test set
    seed(random_seed + 1)
    Z = utils.sample_features(alphas, nmodels)
    estimators = utils.fit(XLS, yLS, Z, model_package)
    YVSpred = utils.predict(estimators, Z, XVS)

    if isClassification:
        YVSpred = np.reshape(le.inverse_transform(np.reshape(YVSpred, -1)), (nmodels, -1))

    yVSpred = model_package['ensemble_prediction'](YVSpred)

    if XTS is not None:
        YTSpred = utils.predict(estimators, Z, XTS)

        if isClassification:
            YTSpred = np.reshape(le.inverse_transform(np.reshape(YTSpred, -1)), (nmodels, -1))

        yTSpred = model_package['ensemble_prediction'](YTSpred)
    else:
        yTSpred = np.nan

    return alphas, objective_values_LS, objective_values_VS, train_indices, yVSpred, yTSpred


def optimise_probabilities(XLS, yLS, XVS, yVS, model_package, alphas_init, lambda_reg, nmodels, batch_size, train_size, nepochs, learning_rate, rho1, rho2, random_seed, eps=1e-8):
    '''
    PRS training.
    '''

    # Set the random seed
    seed(random_seed)

    (nsamples, nfeat) = XLS.shape

    nsplits = int(np.round(nsamples / batch_size))

    beta_min = 1 / nmodels

    alphas = np.zeros((nepochs+1, nfeat))
    objective_values_LS = np.zeros(nepochs+1)
    objective_values_VS = np.zeros(nepochs+1)
    train_indices = [0]

    alphas[0, :] = alphas_init
    alphas_current = alphas[0, :]
    betas = alphas_current.copy()
    p_ratio = 1

    idxperm = permutation(nsamples)
    XLS = XLS[idxperm, :]
    yLS = yLS[idxperm]
    mb_data = list()
    ypred_or_class_proba_mb = list()
    kf = model_package['KFold'](n_splits=nsplits, shuffle=False)
    lossLS = 0
    lossVS = 0

    for k, (train_index, test_index) in enumerate(kf.split(XLS, yLS)):

        # Sample feature subsets
        Z = utils.sample_features(alphas_current, nmodels)

        # Feature that have been selected for at least one model
        Z_selected = np.sum(Z, axis=0)
        feat_idx_to_optimise = np.where(Z_selected > 0)[0]

        # Learn new models and predict samples in mini-batch
        if train_size is not None:
            train_index = choice(train_index, size=train_size, replace=False)

        Xtrain, Xmb = XLS[train_index], XLS[test_index]
        ytrain, ymb = yLS[train_index], yLS[test_index]

        estimators = utils.fit(Xtrain, ytrain, Z, model_package)
        Ypredmb = utils.predict(estimators, Z, Xmb)
        YpredVS = utils.predict(estimators, Z, XVS)

        mb_data.append((ymb, Ypredmb, YpredVS, Z, feat_idx_to_optimise))

        # Initial value of the objective function, when alphas are not optimised yet
        ypred_or_class_proba_mb.append(model_package['ensemble_aggregate'](Ypredmb, p_ratio, model_package['nclasses']))
        ypred_or_class_proba_VS = model_package['ensemble_aggregate'](YpredVS, p_ratio, model_package['nclasses'])
        lossLS += model_package['loss_metric'](ymb, ypred_or_class_proba_mb[k])
        lossVS += model_package['loss_metric'](yVS, ypred_or_class_proba_VS)

    objective_values_LS[0] = lossLS / nsplits + lambda_reg * sum(alphas_current)
    objective_values_VS[0] = lossVS / nsplits + lambda_reg * sum(alphas_current)

    n = 0  # epoch number
    t = 0  # step number

    first_moment = np.zeros(nfeat)
    second_moment = np.zeros(nfeat)

    while n < nepochs:

        n += 1
        lossLS = 0
        lossVS = 0

        for k, (ymb, Ypredmb, YpredVS, Z, feat_idx_to_optimise) in enumerate(mb_data):

            t += 1

            grad = gradient(ymb, ypred_or_class_proba_mb[k], Ypredmb, Z, betas, p_ratio, lambda_reg, feat_idx_to_optimise, model_package)

            # Estimation of the projected gradient
            betas_new = np.clip(betas - eps * grad, a_min=0, a_max=1)
            grad_proj = (betas - betas_new) / eps

            # Adam optimization step
            first_moment = rho1 * first_moment + (1 - rho1) * grad_proj
            second_moment = rho2 * second_moment + (1 - rho2) * grad_proj ** 2
            first_moment_hat = first_moment / (1 - rho1 ** t)
            second_moment_hat = second_moment / (1 - rho2 ** t)
            betas = betas - learning_rate * first_moment_hat / (np.sqrt(second_moment_hat) + eps)

            # Final projection (to correct for numerical errors)
            betas = np.clip(betas, a_min=0, a_max=1)

            # Avoid the critical situation where all the betas drop to zeros
            # (Usually due to the fact that the learning rate is too high)
            if sum(betas) == 0:
                print('!!!Reinializating alphas!!!')
                betas = np.zeros_like(betas) + beta_min

            p_ratio = compute_p_ratio(alphas_current, betas, Z)
            ypred_or_class_proba_mb[k] = model_package['ensemble_aggregate'](Ypredmb, p_ratio, model_package['nclasses'])
            ypred_or_class_proba_VS = model_package['ensemble_aggregate'](YpredVS, p_ratio, model_package['nclasses'])
            lossLS += model_package['loss_metric'](ymb, ypred_or_class_proba_mb[k])
            lossVS += model_package['loss_metric'](yVS, ypred_or_class_proba_VS)

        objective_values_LS[n] = lossLS / nsplits + lambda_reg * sum(betas)
        objective_values_VS[n] = lossVS / nsplits + lambda_reg * sum(betas)
        alphas[n, :] = betas

        nmodels_effective = 0
        denom = sum(p_ratio ** 2)
        if denom > 0:
            nmodels_effective = sum(p_ratio) ** 2 / denom

        if nmodels_effective < 0.9 * nmodels:

            alphas_current = betas.copy()
            p_ratio = 1

            # Train new models

            train_indices.append(n)

            idxperm = permutation(nsamples)
            XLS = XLS[idxperm, :]
            yLS = yLS[idxperm]
            mb_data = list()
            kf = model_package['KFold'](n_splits=nsplits, shuffle=False)
            for k, (train_index, test_index) in enumerate(kf.split(XLS, yLS)):

                # Sample feature subsets
                Z = utils.sample_features(alphas_current, nmodels)

                # Feature that have been selected for at least one model
                Z_selected = np.sum(Z, axis=0)
                feat_idx_to_optimise = np.where(Z_selected > 0)[0]

                # Learn new models and predict samples in mini-batch
                if train_size is not None:
                    train_index = choice(train_index, size=train_size, replace=False)

                Xtrain, Xmb = XLS[train_index], XLS[test_index]
                ytrain, ymb = yLS[train_index], yLS[test_index]

                estimators = utils.fit(Xtrain, ytrain, Z, model_package)
                Ypredmb = utils.predict(estimators, Z, Xmb)
                YpredVS = utils.predict(estimators, Z, XVS)

                mb_data.append((ymb, Ypredmb, YpredVS, Z, feat_idx_to_optimise))
                ypred_or_class_proba_mb[k] = model_package['ensemble_aggregate'](Ypredmb, p_ratio, model_package['nclasses'])

    idx_best = np.argmin(objective_values_VS)

    return alphas[idx_best], objective_values_LS, objective_values_VS, train_indices


def compute_p_ratio(alphas, betas, Z, eps=1e-6):
    """
    :param alphas:
    :param betas:
    :param Z:
    :return:
    """

    nmodels = Z.shape[0]

    alphas = np.tile(alphas, (nmodels, 1))
    betas = np.tile(betas, (nmodels, 1))

    alphas[~Z] = 1 - alphas[~Z]
    betas[~Z] = 1 - betas[~Z]

    alphas = np.clip(alphas, a_min=eps, a_max=None)

    p_ratio = np.prod(betas/alphas, axis=1)
    p_ratio = np.clip(p_ratio, 0, 1e50)  # To avoid overflows

    return p_ratio


def gradient(y, ypred_or_class_proba, Ypred, Z, betas, p_ratio, lambda_reg, feat_idx_to_optimise, model_package, eps=1e-6):
    """
    Computes the gradient of the objective function w.r.t. alphas
    """

    (nmodels, nfeat) = Z.shape

    # Score function
    h = np.tile(betas[feat_idx_to_optimise], (nmodels, 1))
    h[~Z[:, feat_idx_to_optimise]] = h[~Z[:, feat_idx_to_optimise]] - 1

    # Positive values too close to zero are set to eps
    h[Z[:, feat_idx_to_optimise]] = np.clip(h[Z[:, feat_idx_to_optimise]], a_min=eps, a_max=None)
    # Negative values too close to zero are set to -eps
    h[~Z[:, feat_idx_to_optimise]] = np.clip(h[~Z[:, feat_idx_to_optimise]], a_min=None, a_max=-eps)

    h = 1 / h

    grad_output = model_package['expected_output_gradient'](y, Ypred, h, p_ratio, model_package['nclasses'], model_package['variance_reduction'])

    grad = np.zeros(nfeat)
    grad[feat_idx_to_optimise] = model_package['loss_gradient'](y, ypred_or_class_proba, grad_output) + lambda_reg

    return grad
