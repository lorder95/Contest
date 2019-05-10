import pickle


def laod_test_pk(org=False):
    if org:
        file = open('./data/test_or.pk', 'rb')
    else:
        file = open('./data/test.pk', 'rb')
    (X_train, y_train) = pickle.load(file)
    file.close()

    return X_train, y_train


def load_pk(org=False):
    file = open('./data/train.pk', 'rb')
    (X_train, y_train) = pickle.load(file)
    file.close()

    if org:
        file = open('./data/validation_or.pk', 'rb')
    else:
        file = open('./data/validation.pk', 'rb')
    (X_val, y_val) = pickle.load(file)
    file.close()

    return X_train, y_train, X_val, y_val
