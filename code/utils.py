import numpy as np
from sklearn.metrics import accuracy_score
from numpy.random import rand
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from multiprocessing import Pool

def get_estimator(base_model, isClassification):

    estimator_kwargs = dict()

    if isClassification:
        if base_model == 'kNN':
            estimator = KNeighborsClassifier
        elif base_model == 'SVM':
            estimator = SVC
        elif base_model == 'tree':
            estimator = DecisionTreeClassifier
        else:
            raise ValueError('Error base model.')
    else:
        if base_model == 'kNN':
            estimator = KNeighborsRegressor
        elif base_model == 'SVM':
            estimator = SVR
        elif base_model == 'tree':
            estimator = DecisionTreeRegressor
        else:
            raise ValueError('Error model class.')

    return estimator, estimator_kwargs


def ensemble_average(Ypred, p_ratio, nclasses):
    """
    :param Ypred: array of shape (nmodels, nsamples)
    :param p_ratio: pbeta/palpha for importance sampling
    :param nclasses: only here for compatibility purpose
    :return: array (nsamples,) containing the ensemble average
    """

    return np.mean(np.reshape(p_ratio, (-1, 1)) * Ypred, axis=0)


def class_probability(Ypred, p_ratio, nclasses):
    """
    :param Ypred: array of shape (nmodels, nsamples)
    :param p_ratio: pbeta/palpha for importance sampling
    :param nclasses: number of possible classes
    :return: array (nsamples, nclasses) containing the class probabilities
    """

    (nmodels, nsamples) = Ypred.shape

    # Class probas for all the classes except one
    class_indicator = np.zeros((nmodels, nsamples, nclasses-1))
    for k in list(range(1, nclasses)):
        class_indicator[:, :, k-1] = Ypred == k

    class_proba = np.zeros((nsamples, nclasses))
    class_proba[:, 1:] = np.mean(np.reshape(p_ratio, (-1, 1, 1)) * class_indicator, axis=0)

    # Probability of the remaining class
    class_proba[:, 0] = 1 - np.sum(class_proba[:, 1:], axis=1)

    return class_proba


def misclassification_rate(ytrue, ypred):

    return 1 - accuracy_score(ytrue, ypred)


def cross_entropy(ytrue, class_proba, eps=1e-6):
    """
    :param ytrue: true output values.
    :param class_proba: array of shape (nsamples, nclasses) with class probabilities
    :return: cross entropy loss
    """

    true_class_proba = [class_proba[i, k] for i, k in enumerate(ytrue)]
    true_class_proba = np.clip(true_class_proba, eps, 1)

    return -np.mean(np.log(true_class_proba))


def ensemble_prediction_classification(Ypred):

    nsamples = Ypred.shape[1]
    ypred = np.zeros(nsamples, dtype=Ypred.dtype)
    for i in range(nsamples):
        classes, counts = np.unique(Ypred[:, i], return_counts=True)
        ypred[i] = classes[counts.argmax()]

    return ypred


def ensemble_prediction_classification_rase(Ypred, cutoff):

    prop_pred1 = np.mean(Ypred, axis=0)
    ypred = np.zeros(len(prop_pred1), dtype=int)
    ypred[prop_pred1 > cutoff] = 1

    return ypred


def ensemble_prediction_regression(Ypred):

    return np.mean(Ypred, axis=0)


def sample_features(alphas, nmodels):
    '''
    Feature sampling
    '''

    nfeat = len(alphas)
    Z = rand(nmodels, nfeat) < alphas
    return Z


def learn_models(XLS, yLS, XTS, Z, model_package):
    """
    For each subset of selected features in Z, learn a model and return predictions on test set.
    """

    nmodels = Z.shape[0]

    # Check if at least one feature is selected for each model
    Z_selected = np.sum(Z, axis=1)
    idx_none = np.where(Z_selected == 0)[0]
    idx_selected = np.where(Z_selected > 0)[0]

    YTSpred = np.zeros((nmodels, XTS.shape[0]), dtype=yLS.dtype)

    # Models with no selected feature
    # Regression: these models output the average output in the learning sample
    # Classification: these models output the majority class in the learning sample
    estimator = model_package['constant_estimator']
    estimator.fit(XLS, yLS)
    YTSpred[idx_none, :] = estimator.predict(XTS)

    # Models with at least one selected feature
    for t in idx_selected:
        YTSpred[t, :] = learn_one_model(t, XLS, yLS, XTS, Z, model_package)

    return YTSpred


def learn_one_model(t, XLS, yLS, XTS, Z, model_package):
    """
    Learn a model and return predictions on test set
    """

    # Random feature subspace
    XLS_subspace = XLS[:, Z[t, :]]
    XTS_subspace = XTS[:, Z[t, :]]

    estimator = model_package['model'](**model_package['model_kwargs'])
    estimator.fit(XLS_subspace, yLS)

    yTSpred = estimator.predict(XTS_subspace)

    return yTSpred


def learn_multi_models(dataLS, dataTS, input_indices, Z, model_package):

    """
    Learn and predict multiple ensemble models.
    """

    (nsamplesTS, nfeat) = dataTS.shape
    nmodels = Z.shape[0]

    dataTS_pred = np.zeros((nmodels, nsamplesTS, nfeat), dtype=dataLS.dtype)

    for j in range(nfeat):
        dataTS_pred[:, :, j] = learn_models(dataLS[:, input_indices], dataLS[:, j], dataTS[:, input_indices], Z[:, :, j], model_package)

    return dataTS_pred


def network_fit(dataLS, input_indices, Z, model_package):

    estimators = list()

    for j in range(dataLS.shape[1]):
        estimators_j = fit(dataLS[:, input_indices], dataLS[:, j], Z[:, :, j], model_package)
        estimators.append(estimators_j)

    return estimators


def network_predict(estimators, input_indices, Z, dataTS):

    (nsamplesTS, nfeat) = dataTS.shape
    nmodels = Z.shape[0]

    dataTS_pred = np.zeros((nmodels, nsamplesTS, nfeat), dtype=dataTS.dtype)

    for j, estimators_j in enumerate(estimators):
        dataTS_pred[:, :, j] = predict(estimators_j, Z[:, :, j], dataTS[:, input_indices])

    return dataTS_pred


def fit(XLS, yLS, Z, model_package, nthreads=1):
    """
    For each subset of selected features in Z, learn a model and return predictions on test set.
    """

    nmodels = Z.shape[0]

    # Model with no selected feature
    # Regression: this model outputs the average output in the learning sample
    # Classification: this model outputs the majority class in the learning sample
    constant_estimator = model_package['constant_estimator']
    constant_estimator.fit(XLS, yLS)

    if nthreads > 1:

        input_data = list()

        for t in range(nmodels):
            input_data.append([t, XLS, yLS, Z, constant_estimator, model_package])

        with Pool(nthreads) as pool:
            estimators = pool.starmap(fit_one_estimator, input_data)

    else:

        estimators = list()

        for t in range(nmodels):

            estimator = fit_one_estimator(t, XLS, yLS, Z, constant_estimator, model_package)
            estimators.append(estimator)

    return estimators


def fit_one_estimator(t, XLS, yLS, Z, constant_estimator, model_package):

    if np.sum(Z[t, :]) == 0:

        return constant_estimator

    else:

        # Random feature subspace
        XLS_subspace = XLS[:, Z[t, :]]

        estimator = model_package['model'](**model_package['model_kwargs'])
        estimator.fit(XLS_subspace, yLS)

        return estimator


def predict(estimators, Z, XTS, nthreads=1):

    if nthreads > 1:

        nestimators = len(estimators)

        input_data = list()
        for t in range(nestimators):
            input_data.append([estimators[t], XTS[:, Z[t, :]]])

        with Pool(nthreads) as pool:
            all_outputs = pool.starmap(predict_one_estimator, input_data)

        YTSpred = np.zeros((nestimators, XTS.shape[0]), dtype=all_outputs[0].dtype)
        for t, yTSpred in enumerate(all_outputs):
            YTSpred[t, :] = yTSpred

    else:

        nestimators = len(estimators)

        yTSpred = predict_one_estimator(estimators[0], XTS[:, Z[0, :]])

        YTSpred = np.zeros((nestimators, XTS.shape[0]), dtype=yTSpred.dtype)
        YTSpred[0, :] = yTSpred

        for t in range(1, nestimators):
            YTSpred[t, :] = predict_one_estimator(estimators[t], XTS[:, Z[t, :]])

    return YTSpred


def predict_one_estimator(estimator, XTS_subspace):

    return estimator.predict(XTS_subspace)


def decision_cutoff(estimators, Z, XLS, yLS):

    """
    Optimize decision threshold for an ensemble of binary classifiers (RaSE approach).
    """

    p0 = sum(yLS == 0) / len(yLS)
    p1 = sum(yLS == 1) / len(yLS)

    YLSpred = predict(estimators, Z, XLS)
    prop_pred1 = np.mean(YLSpred, axis=0)
    cutoffs = np.unique(prop_pred1)

    best_cutoff = np.nan
    error_min = np.inf

    for cutoff in cutoffs:

        pred0 = prop_pred1 <= cutoff
        error0 = 1 - np.mean(pred0[yLS == 0])
        error1 = np.mean(pred0[yLS == 1])
        error = p0 * error0 + p1 * error1

        if error < error_min:
            error_min = error
            best_cutoff = cutoff

    return best_cutoff



def grad_loss_regression(y, ypred, grad_output):

    return - 2 * np.mean((y.reshape(-1, 1) - ypred.reshape(-1, 1)) * grad_output, axis=0)


def grad_loss_classification(y, class_proba, grad_output, eps=1e-6):
    """"
    :param y: true labels
    :param class_proba:
    :param grad_output:
    :param eps:
    :return:
    """

    true_class_proba = [class_proba[i, k] for i, k in enumerate(y)]
    true_class_proba = np.clip(true_class_proba, eps, 1)

    return -np.mean(grad_output / true_class_proba.reshape(-1, 1), axis=0)


def grad_expected_output_regression(y, Ypred, h, p_ratio, nclasses, variance_reduction, eps=1e-6):
    """
    :param y: only here for compatibility purpose
    :param Ypred: (nmodels, nsamples)
    :param h: score function (nmodels, nfeat_to_optimise)
    :param p_ratio:
    :param nclasses: only here for compatibility purpose
    :param variance_reduction: boolean indicating whether or not to apply the variance reduction technique with baseline
    :return: gradient of expected output
    """

    nfeat_to_optimise = h.shape[1]
    nsamples = Ypred.shape[1]

    Ypred = np.transpose(np.tile(Ypred, (nfeat_to_optimise, 1, 1)), axes=(1, 2, 0))

    if variance_reduction:

        h_square = h ** 2
        h_var = np.mean(np.reshape(p_ratio, (-1, 1)) * h_square, axis=0)
        h_var = np.clip(h_var, a_min=eps, a_max=None)

        h_square = np.transpose(np.tile(h_square, (nsamples, 1, 1)), axes=(1, 0, 2))

        b = np.mean(np.reshape(p_ratio, (-1, 1, 1)) * Ypred * h_square, axis=0) / h_var.reshape(1, -1)

    else:

        b = 0

    h = np.transpose(np.tile(h, (nsamples, 1, 1)), axes=(1, 0, 2))

    return np.mean(np.reshape(p_ratio, (-1, 1, 1)) * (Ypred - b) * h, axis=0)


def grad_expected_output_classification(y, Ypred, h, p_ratio, nclasses, variance_reduction, eps=1e-6):
    """
    :param y: true labels (nsamples,)
    :param Ypred: (nmodels, nsamples)
    :param h: score function (nmodels, nfeat_to_optimise)
    :param p_ratio:
    :param nclasses:
    :param variance_reduction: boolean indicating whether or not to apply the variance reduction technique with baseline
    :return: gradient of true class probability
    """

    nfeat_to_optimise = h.shape[1]
    (nmodels, nsamples) = Ypred.shape


    # Gradient of class probability for all the classes except one
    class_indicator = np.zeros((nmodels, nsamples, nclasses-1))

    for k in list(range(1, nclasses)):
        class_indicator[:, :, k-1] = Ypred == k

    class_indicator = np.transpose(np.tile(class_indicator, (nfeat_to_optimise, 1, 1, 1)), axes=(1, 2, 0, 3))

    if variance_reduction:
        h_square = h ** 2

        h_var = np.mean(np.reshape(p_ratio, (-1, 1)) * h_square, axis=0)
        h_var = np.clip(h_var, a_min=eps, a_max=None)

        h_square = np.tile(h_square, (nsamples, 1, 1))
        h_square = np.transpose(np.tile(h_square, (nclasses - 1, 1, 1, 1)), axes=(2, 1, 3, 0))

        b = np.mean(np.reshape(p_ratio, (-1, 1, 1, 1)) * class_indicator * h_square, axis=0) / np.reshape(h_var, (1, -1, 1))
    else:
        b = 0

    h = np.tile(h, (nsamples, 1, 1))
    h = np.transpose(np.tile(h, (nclasses - 1, 1, 1, 1)), axes=(2, 1, 3, 0))

    grad_class_proba = np.zeros((nsamples, nfeat_to_optimise, nclasses))
    grad_class_proba[:, :, 1:] = np.mean(np.reshape(p_ratio, (-1, 1, 1, 1)) * (class_indicator - b) * h, axis=0)
    # Sum of gradients must be zero
    grad_class_proba[:, :, 0] = -np.sum(grad_class_proba[:, :, 1:], axis=2)

    grad_true_class_proba = np.zeros((nsamples, nfeat_to_optimise))
    for i, k in enumerate(y):
        grad_true_class_proba[i, :] = grad_class_proba[i, :, k]

    return grad_true_class_proba
