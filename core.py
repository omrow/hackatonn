import warnings
import numpy as np
from sklearn import linear_model, svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn import metrics, cross_validation
    from sklearn.metrics.base import UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def test_different_classifiers(train_authors, train_data, folds=10, shuffle_=False):
    train_data = np.asarray(train_data)
    train_authors = np.asarray(train_authors)
    kf = cross_validation.StratifiedKFold(train_authors, n_folds=folds, shuffle=shuffle_)
    names = ["Logistic regression", "Support vector machine", "Gaussian Naive Bayes", "Decision tree", "Random forest"]
    classifiers = [linear_model.LogisticRegression(), svm.SVC(), GaussianNB(), tree.DecisionTreeClassifier(max_depth=None, min_samples_split=1), RandomForestClassifier(n_estimators=30, max_depth=None, min_samples_split=1)]
    for name, clf in zip(names, classifiers):
        results = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'cm': []}
        for train_index, test_index in kf:
            X_train, X_test = train_data[train_index], train_data[test_index]
            y_train, y_test = train_authors[train_index], train_authors[test_index]
            clf = clf.fit(X_train, y_train)
            predicted = clf.predict(X_test)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                results['acc'].append(metrics.accuracy_score(y_test, predicted))
                results['prec'].append(metrics.precision_score(y_test, predicted, average='weighted'))
                results['rec'].append(metrics.recall_score(y_test, predicted, average='weighted'))
                results['f1'].append(metrics.f1_score(y_test, predicted, average='weighted'))
        print (name, ': Accuracy: %s, F1-measure: %s, Predicted: %s' % (np.mean(results['acc']), np.mean(results['f1']), np.mean(results['prec'])))