from multiprocessing import Pool
import numpy as np
from numpy.random import seed, permutation, choice
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
import utils
from sklearn.dummy import DummyRegressor, DummyClassifier


def rase(XLS, yLS, XVS, yVS, estimator, estimator_kwargs, isClassification, nmodels=100, B=500, D=None, niterations=10,
         C0=0.1, XTS=None, normalize_data=True, tune_decision_cutoff=False, random_seed=100, nthreads=1):
    '''
        :param XLS: array of shape (n_samples_LS, n_features) containing the inputs of the learning set.
        :param yLS: array of shape (n_samples_LS,) containing the outputs of the learning set (class labels in classification, real numbers in regression).
        :param XVS: array of shape (n_samples_VS, n_features) containing the inputs of the validation set.
        :param yVS: array of shape (n_samples_VS,) containing the outputs of the validation set (class labels in classification, real numbers in regression).
        :param estimator: scikit-learn learner class.
        :param estimator_kwargs: dictionary containing the hyper-parameters of the estimator.
        :param isClassification: boolean indicating if this is a classification problem or not (otherwise it is a regression problem)
        :param nmodels: number of base models in the ensemble.
        :param B: number of subspace candidates generated for each base model.
        :param D: maximal subspace size when generating random subspaces. If None, D is set to min(sqrt(n_samples_LS), nfeatures).
        :param niterations: number of iterations.
        :param C0: positive constant used to set the minimum feature selection probability.
        :param XTS: array of shape (n_samples_TS, nfeatures) containing the inputs of the test set.
        :param normalize_data: boolean indicating if input data must be normalized.
        :param tune_decision_cutoff: boolean indicating if the decision cutoff must be tuned like in the original RaSE method (only for classification)
        :param random_seed: random seed
        :param nthreads: number of threads used for parallel computing.
        :return: a tuple (feat_importances, feat_subset_sizes, yTSpred)
        - feat_importances: array of shape (nfeatures,) containing the feature importances.
        - feat_subspace_sizes: array of shape (nmodels,) containing the subspace sizes (i.e. the number of features for each base model) in the trained ensemble.
        - yTSpred: array of shape (nsamplesTS,) containing the predictions of the trained ensemble on the test set.
    '''

    (nsamples, nfeat) = XLS.shape

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

    if isClassification:
        # Classification

        # Transform labels into integers
        le = LabelEncoder()
        yLS = le.fit_transform(yLS)

        model_package['constant_estimator'] = DummyClassifier(strategy='most_frequent')
        model_package['KFold'] = StratifiedKFold
        if tune_decision_cutoff:
            model_package['ensemble_prediction'] = utils.ensemble_prediction_classification_rase
        else:
            model_package['ensemble_prediction'] = utils.ensemble_prediction_classification
        model_package['error_metric'] = utils.misclassification_rate

    else:
        # Regression
        model_package['constant_estimator'] = DummyRegressor(strategy='mean')
        model_package['KFold'] = KFold
        model_package['ensemble_prediction'] = utils.ensemble_prediction_regression
        model_package['error_metric'] = mean_squared_error

    if D is None:
        D = min(np.floor(np.sqrt(nsamples)), nfeat)

    # Initialize at uniform distribution
    feat_probas = np.ones(nfeat) / nfeat
    errorVS_min = np.inf
    estimators_best = np.nan
    Z_best = np.nan
    feat_importances_best = np.nan
    feat_subspace_sizes_best = np.nan

    for k in range(niterations):

        # Feature subspaces
        Z = np.zeros((nmodels, nfeat), dtype=bool)

        if nthreads > 1:
            input_data = list()
            for t in range(nmodels):
                input_data.append([XLS, yLS, B, D, feat_probas, model_package, random_seed * k + t])

            with Pool(nthreads) as pool:
                alloutput = pool.starmap(get_best_model, input_data)

            for t, feat_subset in enumerate(alloutput):
                Z[t, :] = feat_subset

        else:
            for t in range(nmodels):
                Z[t, :] = get_best_model(XLS, yLS, B, D, feat_probas, model_package, random_seed * k + t)

        # Feature importances
        feat_importances = np.mean(Z, axis=0)
        feat_subspace_sizes = np.sum(Z, axis=1)

        # Update feature selection probabilities
        idx = feat_importances > C0 / np.log(nfeat)
        feat_probas = np.zeros(nfeat)
        feat_probas[idx] = feat_importances[idx]
        feat_probas[~idx] = C0 / nfeat
        feat_probas = feat_probas / sum(feat_probas)

        # Train ensemble and predict on validation set
        estimators = utils.fit(XLS, yLS, Z, model_package)
        YVSpred = utils.predict(estimators, Z, XVS)

        if tune_decision_cutoff:
            cutoff = utils.decision_cutoff(estimators, Z, XLS, yLS)
            yVSpred = model_package['ensemble_prediction'](YVSpred, cutoff)
        else:
            yVSpred = model_package['ensemble_prediction'](YVSpred)

        if isClassification:
            yVSpred = le.inverse_transform(yVSpred)

        errorVS = model_package['error_metric'](yVS, yVSpred)
        if errorVS < errorVS_min:
            errorVS_min = errorVS
            estimators_best = estimators
            Z_best = Z
            feat_importances_best = feat_importances
            feat_subspace_sizes_best = feat_subspace_sizes

    if XTS is not None:
        YTSpred = utils.predict(estimators_best, Z_best, XTS)
        if tune_decision_cutoff:
            yTSpred = model_package['ensemble_prediction'](YTSpred, cutoff)
        else:
            yTSpred = model_package['ensemble_prediction'](YTSpred)
        if isClassification:
            yTSpred = le.inverse_transform(yTSpred)
    else:
        yTSpred = np.nan

    return feat_importances_best, feat_subspace_sizes_best, yTSpred


def get_best_model(XLS, yLS, B, D, feat_probas, model_package, random_seed):

    # Set the random seed
    seed(random_seed)

    nsamples, nfeat = XLS.shape

    dmax = sum(feat_probas > 0)

    # Shuffle samples
    idxperm = permutation(nsamples)
    XLS = XLS[idxperm, :]
    yLS = yLS[idxperm]

    Z = np.zeros((B, nfeat), dtype=bool)

    # Sample feature subsets
    for t in range(B):
        # Sample subset size
        d = min(int(choice(np.arange(1, D + 1), size=1)[0]), dmax)

        # Sample features
        idx = choice(nfeat, size=d, replace=False, p=feat_probas)
        Z[t, idx] = True

    # Use CV to select best feature subset
    nsplits = 10
    kf = model_package['KFold'](n_splits=nsplits, shuffle=False)
    errors = np.zeros((nsplits, B))

    for k, (train_index, test_index) in enumerate(kf.split(XLS, yLS)):

        Xtrain, Xtest = XLS[train_index], XLS[test_index]
        ytrain, ytest = yLS[train_index], yLS[test_index]

        Ypred = utils.learn_models(Xtrain, ytrain, Xtest, Z, model_package)

        for t in range(B):
            errors[k, t] = model_package['error_metric'](ytest, Ypred[t, :])

    return Z[np.argmin(np.mean(errors, axis=0))]
