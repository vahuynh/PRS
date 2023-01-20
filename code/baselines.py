import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,\
    GradientBoostingRegressor, GradientBoostingClassifier


def RF(XLS, yLS, XVS, yVS, XTS, isClassification, nmodels=100, normalize_data=True):
    '''
    Random forest, with optimization of the hyper-parameter K (i.e. the number of features sampled at each tree node)
    on the validation set.

    :param XLS: array of shape (n_samples_LS, n_features) containing the inputs of the learning set.
    :param yLS: array of shape (n_samples_LS,) containing the outputs of the learning set (class labels in classification, real numbers in regression).
    :param XVS: array of shape (n_samples_VS, n_features) containing the inputs of the validation set.
    :param yVS: array of shape (n_samples_VS,) containing the outputs of the validation set (class labels in classification, real numbers in regression).
    :param XTS: array of shape (n_samples_TS, n_features) containing the inputs of the test set.
    :param isClassification: boolean indicating if this is a classification problem or not (otherwise it is a regression problem)
    :param nmodels: number of trees in the Random forest ensemble.
    :param normalize_data: boolean indicating if input data must be normalized.

    returns: a tuple (selected_K, yTSpred, feat_importances)
    - selected_K: optimized value of K
    - yTSpred: array of shape (n_samples_TS,) containing the predictions of the trained RS model on the test set.
    - feat_importances: feature importances
    '''

    nfeat = XLS.shape[1]

    # Values of K to test
    Ksqrt = int(np.round(np.sqrt(nfeat)))
    K_to_test = [1, Ksqrt, nfeat]
    for div in [100, 50, 20, 10, 5, 3, 2]:
        K_to_test.append(int(np.round(nfeat / div)))
    K_to_test = np.array(K_to_test)
    K_to_test = K_to_test[K_to_test > 0]

    if normalize_data:
        X_scaler = StandardScaler()
        XLS = X_scaler.fit_transform(XLS)
        XTS = X_scaler.transform(XTS)
        XVS = X_scaler.transform(XVS)

    errorVS_min = np.inf
    selected_K = np.nan
    yTSpred = np.nan
    feat_importances = np.nan

    if isClassification:
        RF_estimator = RandomForestClassifier
    else:
        RF_estimator = RandomForestRegressor

    # Select the value of K using a validation set
    for K in K_to_test:

        estimator = RF_estimator(n_estimators=nmodels, max_features=int(K))
        estimator.fit(XLS, yLS)
        yVSpred = estimator.predict(XVS)

        if isClassification:
            errorVS = 1 - accuracy_score(yVS, yVSpred)
        else:
            errorVS = mean_squared_error(yVS, yVSpred)

        if errorVS < errorVS_min:
            errorVS_min = errorVS
            selected_K = K
            yTSpred = estimator.predict(XTS)
            feat_importances = estimator.feature_importances_

    return (selected_K, yTSpred, feat_importances)


def GBDT(XLS, yLS, XVS, yVS, XTS, isClassification, nmodels=100, normalize_data=True):
    '''
    Random forest, with optimization of the hyper-parameter max_depth (i.e. the maximal tree depth)
    on the validation set.

    :param XLS: array of shape (n_samples_LS, n_features) containing the inputs of the learning set.
    :param yLS: array of shape (n_samples_LS,) containing the outputs of the learning set (class labels in classification, real numbers in regression).
    :param XVS: array of shape (n_samples_VS, n_features) containing the inputs of the validation set.
    :param yVS: array of shape (n_samples_VS,) containing the outputs of the validation set (class labels in classification, real numbers in regression).
    :param XTS: array of shape (n_samples_TS, n_features) containing the inputs of the test set.
    :param isClassification: boolean indicating if this is a classification problem or not (otherwise it is a regression problem)
    :param nmodels: number of trees in the ensemble.
    :param normalize_data: boolean indicating if input data must be normalized.

    returns: a tuple (selected_max_depth, yTSpred, feat_importances)
    - selected_max_depth: optimized value of max_depth
    - yTSpred: array of shape (n_samples_TS,) containing the predictions of the trained RS model on the test set.
    - feat_importances: feature importances
    '''

    if normalize_data:
        # Normalize data
        X_scaler = StandardScaler()
        XLS = X_scaler.fit_transform(XLS)
        XVS = X_scaler.transform(XVS)
        XTS = X_scaler.transform(XTS)

    max_depth_to_test = list(range(1, 11))

    errorVS_min = np.inf
    selected_max_depth = np.nan
    yTSpred = np.nan
    feat_importances = np.nan

    if isClassification:
        boosting_estimator = GradientBoostingClassifier
    else:
        boosting_estimator = GradientBoostingRegressor

    # Select the value of max_depth using a validation set
    for max_depth in max_depth_to_test:

        estimator = boosting_estimator(max_depth=int(max_depth), n_estimators=nmodels)
        estimator.fit(XLS, yLS)
        yVSpred = estimator.predict(XVS)

        if isClassification:
            errorVS = 1 - accuracy_score(yVS, yVSpred)
        else:
            errorVS = mean_squared_error(yVS, yVSpred)

        if errorVS < errorVS_min:
            errorVS_min = errorVS
            selected_max_depth = max_depth
            yTSpred = estimator.predict(XTS)
            feat_importances = estimator.feature_importances_

    return (selected_max_depth, yTSpred, feat_importances)


def RS(XLS, yLS, XVS, yVS, XTS, estimator, estimator_kwargs, isClassification, nmodels=100, normalize_data=True):
    '''
    Random Subspace, with optimization of the hyper-parameter K (i.e. number of features sampled per base model)
    on the validation set.

    :param XLS: array of shape (n_samples_LS, n_features) containing the inputs of the learning set.
    :param yLS: array of shape (n_samples_LS,) containing the outputs of the learning set (class labels in classification, real numbers in regression).
    :param XVS: array of shape (n_samples_VS, n_features) containing the inputs of the validation set.
    :param yVS: array of shape (n_samples_VS,) containing the outputs of the validation set (class labels in classification, real numbers in regression).
    :param XTS: array of shape (n_samples_TS, n_features) containing the inputs of the test set.
    :param estimator: scikit-learn learner class.
    :param estimator_kwargs: dictionary containing the hyper-parameters of the estimator.
    :param isClassification: boolean indicating if this is a classification problem or not (otherwise it is a regression problem)
    :param nmodels: number of base models in the ensemble.
    :param normalize_data: boolean indicating if input data must be normalized.

    returns: a tuple (selected_K, yTSpred)
    - selected_K: optimized value of K
    - yTSpred: array of shape (n_samples_TS,) containing the predictions of the trained RS model on the test set.
    '''

    nfeat = XLS.shape[1]

    # Values of K to test
    Ksqrt = int(np.round(np.sqrt(nfeat)))
    K_to_test = [1, Ksqrt, nfeat]
    for div in [100, 50, 20, 10, 5, 3, 2]:
        K_to_test.append(int(np.round(nfeat / div)))
    K_to_test = np.array(K_to_test)
    K_to_test = K_to_test[K_to_test > 0]

    if normalize_data:
        # Normalize data
        X_scaler = StandardScaler()
        XLS = X_scaler.fit_transform(XLS)
        XVS = X_scaler.transform(XVS)
        XTS = X_scaler.transform(XTS)

    errorVS_min = np.inf
    selected_K = np.nan
    yTSpred = np.nan

    # Use validation set to select the value of K
    for K in K_to_test:

        estimators, Z = RS_fit(XLS, yLS, estimator, estimator_kwargs, K, nmodels)
        yVSpred = RS_predict(XVS, estimators, Z, isClassification)

        if isClassification:
            errorVS = 1 - accuracy_score(yVS, yVSpred)
        else:
            errorVS = mean_squared_error(yVS, yVSpred)


        if errorVS < errorVS_min:
            errorVS_min = errorVS
            selected_K = K
            yTSpred = RS_predict(XTS, estimators, Z, isClassification)


    return (selected_K, yTSpred)


def RS_fit(XLS, yLS, estimator, estimator_kwargs, K, nmodels):
    """
    Learn a Random Subspace model

    """

    nfeat = XLS.shape[1]

    estimators = list()
    Z = np.zeros((nmodels, nfeat), dtype=bool)

    for t in range(nmodels):

        # Sample features
        idx_features = resample(np.arange(nfeat), replace=False, n_samples=K)
        Z[t, idx_features] = True

        # Random feature subspace
        XLS_subspace = XLS[:, Z[t, :]]

        est = estimator(**estimator_kwargs)
        est.fit(XLS_subspace, yLS)
        estimators.append(est)

    return estimators, Z


def RS_predict(XTS, estimators, Z, isClassification):
    """
    Predict test set
    """

    nsamples = XTS.shape[0]
    nmodels = Z.shape[0]
    YTSpred = np.zeros((nmodels, nsamples))

    for t, est in enumerate(estimators):

        XTS_subspace = XTS[:, Z[t, :]]
        YTSpred[t, :] = est.predict(XTS_subspace)

    if isClassification:

        yTSpred = np.zeros(nsamples, dtype=YTSpred.dtype)

        for i in range(nsamples):
            classes, counts = np.unique(YTSpred[:, i], return_counts=True)
            yTSpred[i] = classes[counts.argmax()]

    else:

        yTSpred = np.mean(YTSpred, axis=0)

    return yTSpred
