from sklearn.model_selection import train_test_split
import numpy as np
import openml
from sklearn.preprocessing import OneHotEncoder

def tabular_data(dataset_idx, random_state, isClassification, with_categorical_features, LSsize=1000, VSsize=1000, TSsize=1000):
    '''

    Download a tabular dataset and sample training, validation and test sets.

    :param dataset_idx: index of dataset within the benchmark suite.
    :param random_state: integer specifying the random seed for the data sampling.
    :param isClassification: boolean specifying whether it is a classification problem (otherwise it is a regression problem).
    :param with_categorical_features: boolean specifying whether we want the benchmark with datasets that include categorical features.
    :param LSsize: number of samples in the training set.
    :param VSsize: number of samples in the validation set.
    :param TSsize: number of samples in the test set.

    Returns: a tuple (dataset_name, dataset)
    - dataset_name: name of the dataset
    - dataset: dictionary containing the data
        dataset['XLS']: inputs of the training set
        dataset['yLS']: outputs of the training set
        dataset['XVS']: inputs of the validation set
        dataset['yVS']: outputs of the validation set
        dataset['XTS']: inputs of the test set
        dataset['yTS']: outputs of the test set
    '''

    if isClassification:
        if with_categorical_features:
            SUITE_ID = 304  # Classification on numerical and categorical features
        else:
            SUITE_ID = 298  # Classification on numerical features
    else:
        if with_categorical_features:
            SUITE_ID = 299  # Regression on numerical and categorical features
        else:
            SUITE_ID = 297  # Regression on numerical features

    benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite
    task_id = benchmark_suite.tasks[dataset_idx]
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="array",
                                                                    target=dataset.default_target_attribute)

    if with_categorical_features:
        # One-hot encoding of categorical features
        numerical_indicator = [not x for x in categorical_indicator]
        enc = OneHotEncoder(sparse=False)
        Xcat = enc.fit_transform(X[:, categorical_indicator])
        Xnum = X[:, numerical_indicator]
        X = np.concatenate((Xcat, Xnum), axis=1)

    # Split data into train, validation and test sets
    if isClassification:
        XLS, X_tmp, yLS, y_tmp = train_test_split(X, y, train_size=LSsize, random_state=random_state, stratify=y)
        XVS, XTS, yVS, yTS = train_test_split(X_tmp, y_tmp, train_size=VSsize, random_state=random_state, stratify=y_tmp)

        if len(yTS) > TSsize:
            XTS, X_tmp, yTS, y_tmp = train_test_split(XTS, yTS, train_size=TSsize, random_state=random_state, stratify=yTS)
    else:
        XLS, X_tmp, yLS, y_tmp = train_test_split(X, y, train_size=LSsize, random_state=random_state)
        XVS, XTS, yVS, yTS = train_test_split(X_tmp, y_tmp, train_size=VSsize, random_state=random_state)
        XTS = XTS[:TSsize]
        yTS = yTS[:TSsize]

    dataset1 = dict()
    # Training set
    dataset1['XLS'] = XLS
    dataset1['yLS'] = yLS

    # Validation set
    dataset1['XVS'] = XVS
    dataset1['yVS'] = yVS

    # Test set
    dataset1['XTS'] = XTS
    dataset1['yTS'] = yTS

    return dataset.name, dataset1
