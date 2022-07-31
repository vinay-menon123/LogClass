from asyncio.windows_events import NULL
import os
import argparse
from uuid import uuid4
import sys
import functools
import json
import shutil
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import re
import numpy as np
from collections import defaultdict, Counter
from abc import ABC, abstractmethod
from time import time
import pickle
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix

class PUAdapter(object):
    def __init__(self, estimator, hold_out_ratio=0.1, precomputed_kernel=False):
       
        self.estimator = estimator
        self.c = 1.0
        self.hold_out_ratio = hold_out_ratio

        if precomputed_kernel:
            self.fit = self.__fit_precomputed_kernel
        else:
            self.fit = self.__fit_no_precomputed_kernel

        self.estimator_fitted = False

    def __str__(self):
        return 'Estimator:' + str(self.estimator) + '\n' + 'p(s=1|y=1,x) ~= ' + str(self.c) + '\n' + \
            'Fitted: ' + str(self.estimator_fitted)


    def __fit_precomputed_kernel(self, X, y):
        positives = np.where(y == 1.)[0]
        hold_out_size = np.ceil(len(positives) * self.hold_out_ratio)

        if len(positives) <= hold_out_size:
            raise('Not enough positive examples to estimate p(s=1|y=1,x). Need at least ' + str(hold_out_size + 1) + '.')

        np.random.shuffle(positives)
        hold_out = positives[:hold_out_size]

        #Hold out test kernel matrix
        X_test_hold_out = X[hold_out]
        keep = list(set(np.arange(len(y))) - set(hold_out))
        X_test_hold_out = X_test_hold_out[:,keep]

        #New training kernel matrix
        X = X[:, keep]
        X = X[keep]

        y = np.delete(y, hold_out)

        self.estimator.fit(X, y)

        hold_out_predictions = self.estimator.predict_proba(X_test_hold_out)

        try:
            hold_out_predictions = hold_out_predictions[:,1]
        except:
            pass

        c = np.mean(hold_out_predictions)
        self.c = c

        self.estimator_fitted = True


    def __fit_no_precomputed_kernel(self, X, y):
        positives = np.where(y == 1.)[0]
        hold_out_size = np.ceil(len(positives) * self.hold_out_ratio)
        if len(positives) <= hold_out_size:
            raise('Not enough positive examples to estimate p(s=1|y=1,x). Need at least ' + str(hold_out_size + 1) + '.')

        np.random.shuffle(positives)
       # print hold_out_size
        hold_out = positives[:int(hold_out_size)]
        X_hold_out = X[hold_out]
        X = np.delete(X, hold_out,0)
        y = np.delete(y, hold_out)

        self.estimator.fit(X, y)

        hold_out_predictions = self.estimator.predict_proba(X_hold_out)

        try:
            hold_out_predictions = hold_out_predictions[:,1]
        except:
            pass

        c = np.mean(hold_out_predictions)
        self.c = c

        self.estimator_fitted = True

    def predict_proba(self, X):
        if not self.estimator_fitted:
            raise Exception('The estimator must be fitted before calling predict_proba(...).')

        probabilistic_predictions = self.estimator.predict_proba(X)

        try:
            probabilistic_predictions = probabilistic_predictions[:,1]
        except:
            pass

        return probabilistic_predictions / self.c


    def predict(self, X, treshold=0.5):
        if not self.estimator_fitted:
            raise Exception('The estimator must be fitted before calling predict(...).')

        return np.array([1. if p > treshold else -1. for p in self.predict_proba(X)])

def trim(s):
    return s if len(s) <= 80 else s[:77] + "..."


class TestingParameters():
    def __init__(self, params):
        self.params = params
        self.original_state = params['train']

    def __enter__(self):
        self.params['train'] = False

    def __exit__(self, exc_type, exc_value, traceback):
        self.params['train'] = self.original_state


def load_params(params):
    print(params)
    params_file = os.path.join(
        params['id_dir'], f"best_params.json")
    with open(params_file, "r") as fp:
        best_params = json.load(fp)
    params.update(best_params)


def save_params(params):
    params_file = os.path.join(
        params['id_dir'], f"best_params.json")
    with open(params_file, "w") as fp:
        json.dump(params, fp)


def file_handling(params):
    if "raw_logs" in params:
        if not os.path.exists(params['raw_logs']):
            raise FileNotFoundError(
                f"File {params['raw_logs']} doesn't exist. "
                + "Please provide the raw logs path."
            )
        
        logs_directory = os.path.dirname(params['logs'])
        if not os.path.exists(logs_directory):
            os.makedirs(logs_directory)
    else:
        # Checks if preprocessed logs exist as input
        if not os.path.exists(params['logs']):
            raise FileNotFoundError(
                f"File {params['base_dir']} doesn't exist. "
                + "Preprocess target logs first and provide their path."
            )

    if params['train']:
        # Checks if the experiment id already exists
        if os.path.exists(params["id_dir"]) and not params["force"]:
            raise FileExistsError(
                f"directory '{params['id_dir']} already exists. "
                + "Run with --force to overwrite."
                + f"If --force is used, you could lose your training results."
            )
        if os.path.exists(params["id_dir"]):
            shutil.rmtree(params["id_dir"])
        for target_dir in ['id_dir', 'models_dir', 'features_dir']:
            os.makedirs(params[target_dir])
    else:
        # Checks if input models and features are provided
        for concern in ['models_dir', 'features_dir']:
            target_path = params[concern]
            if not os.path.exists(target_path):
                raise FileNotFoundError(
                    "directory '{} doesn't exist. ".format(target_path)
                    + "Run train first before running inference."
                )


def print_params(params):
    print("{:-^80}".format("params"))
    print("Beginning experiment using the following configuration:\n")
    for param, value in params.items():
        print("\t{:>13}: {}".format(param, value))
    print()
    print("-" * 80)


def save_results(results, params):
    df = pd.DataFrame(results)
    file_name = os.path.join(
        params['id_dir'],
        "results.csv",
        )
    df.to_csv(file_name, index=False)

# print-step
def debug(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)           # 3
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")           # 4
        return value
    return wrapper_debug


def print_step(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_print_name(*args, **kwargs):
        print(f"Calling {func.__qualname__}")
        value = func(*args, **kwargs)
        return value
    return wrapper_print_name


def init_main_args():
    """Init command line args used for configuration."""

    parser = argparse.ArgumentParser(
        description="Runs experiment using LogClass Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw_logs",
        metavar="raw_logs",
        type=str,
        nargs=1,
        help="input raw logs file path",
    )
    base_dir_default = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "output"
    )
    parser.add_argument(
        "--base_dir",
        metavar="base_dir",
        type=str,
        nargs=1,
        default=[base_dir_default],
        help="base output directory for pipeline output files",
    )
    parser.add_argument(
        "--logs",
        metavar="logs",
        type=str,
        nargs=1,
        help="input logs file path and output for raw logs preprocessing",
    )
    parser.add_argument(
        "--models_dir",
        metavar="models_dir",
        type=str,
        nargs=1,
        help="trained models input/output directory path",
    )
    parser.add_argument(
        "--features_dir",
        metavar="features_dir",
        type=str,
        nargs=1,
        help="trained features_dir input/output directory path",
    )
    parser.add_argument(
        "--logs_type",
        metavar="logs_type",
        type=str,
        nargs=1,
        default=["open_Apache"],
        choices=[
            "bgl",
            "open_Apache",
            "open_bgl",
            "open_hadoop",
            "open_hdfs",
            "open_hpc",
            "open_proxifier",
            "open_zookeeper",
            ],
        help="Input type of logs.",
    )
    parser.add_argument(
        "--kfold",
        metavar="kfold",
        type=int,
        nargs=1,
        help="kfold crossvalidation",
    )
    parser.add_argument(
        "--healthy_label",
        metavar='healthy_label',
        type=str,
        nargs=1,
        default=["unlabeled"],
        help="the labels of unlabeled logs",
    )
    parser.add_argument(
        "--features",
        metavar="features",
        type=str,
        nargs='+',
        default=["tfilf"],
        choices=["tfidf", "tfilf", "length", "tf"],
        help="Features to be extracted from the logs messages.",
    )
    parser.add_argument(
        "--report",
        metavar="report",
        type=str,
        nargs='+',
        default=["confusion_matrix"],
        choices=["confusion_matrix",
                 "acc",
                 "multi_acc",
                 "top_k_svm",
                 "micro",
                 "macro"
                 ],
        help="Reports to be generated from the model and its predictions.",
    )
    parser.add_argument(
        "--binary_classifier",
        metavar="binary_classifier",
        type=str,
        nargs=1,
        default=["pu_learning"],
        choices=["pu_learning", "regular"],
        help="Binary classifier to be used as anomaly detector.",
    )
    parser.add_argument(
        "--multi_classifier",
        metavar="multi_classifier",
        type=str,
        nargs=1,
        default=["svm"],
        choices=["svm"],
        help="Multi-clas classifier to classify anomalies.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="If set, logclass will train on the given data. Otherwise"
             + "it will run inference on it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force training overwriting previous output with same id.",
    )
    parser.add_argument(
        "--id",
        metavar="id",
        type=str,
        nargs=1,
        help="Experiment id. Automatically generated if not specified.",
    )
    parser.add_argument(
        "--log_group",
        metavar="log_group",
        type=str,
        nargs=1,
        help="Name of log group.",
    )
    parser.add_argument(
        "--swap",
        action="store_true",
        default=False,
        help="Swap testing/training data in kfold cross validation.",
    )

    return parser


def parse_main_args(args):
    """Parse provided args for runtime configuration."""
    params = {
        "report": "macro",
        "train": args.train,
        "force": True,
        "base_dir": args.base_dir[0],
        "logs_type": args.logs_type[0],
        "healthy_label": args.healthy_label[0],
        "features": args.features,
        "binary_classifier": args.binary_classifier[0],
        "multi_classifier": args.multi_classifier[0],
        "swap": args.swap,
        "raw_logs": "LogClass\data\open_source_logs",
    }
    params.update({'id': input("Enter Experiment id: ")})
    params.update({'log_group': input("Enter Log Group Name: ")})
    if args.raw_logs:
        params["raw_logs"] = os.path.normpath(args.raw_logs[0])
    if args.kfold:
        params["kfold"] = args.kfold[0]
    if args.logs:
        params['logs'] = os.path.normpath(args.logs[0])
    else:
        params['logs'] = os.path.join(
            params['base_dir'],
            "preprocessed_logs",
            f"{params['logs_type']}.txt"
        )
    if(params['id'] in dir()):
        print(f"\nExperiment ID: {params['id']}")
    # Creating experiments results folder with the format
    # {experiment_module_name}_{logs_type}_{id}
    experiment_name = os.path.basename(sys.argv[0]).split('.')[0]
    params['id_dir'] = os.path.join(
            params['base_dir'],
            '_'.join((
                experiment_name, params['logs_type'], params['log_group'], params['id']
                ))
        )
    if args.models_dir:
        params['models_dir'] = os.path.normpath(args.models_dir[0])
    else:
        params['models_dir'] = os.path.join(
            params['id_dir'],
            "models",
        )
    if args.features_dir:
        params['features_dir'] = os.path.normpath(args.features_dir[0])
    else:
        params['features_dir'] = os.path.join(
            params['id_dir'],
            "features",
        )
    params['results_dir'] = os.path.join(params['id_dir'], "results")

    return params

class BaseModel(ABC):
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.name = type(model).__name__
        self.train_time = None
        self.run_time = None

    @abstractmethod
    def save(self, **kwargs):
        """
            Abstract method for the subclass to implement how the model is
            saved. Should use the experiment id as reference.
        """
        pass

    @abstractmethod
    def load(self, **kwargs):
        """
            Abstract method for the subclass to implement how it's meant to be
            loaded. Should correspond to how the save method saves the model.
        """
        pass

    @print_step
    def predict(self, X, **kwargs):
        """
            Wraps original model predict and times its running time.
        """
        t0 = time()
        pred = self.model.predict(X, **kwargs)
        t1 = time()
        lapse = t1 - t0
        self.run_time = lapse
        print(f"{self.name} took {lapse}s to run inference.")
        return pred

    @print_step
    def fit(self, X, Y, **kwargs):
        """
            Wraps original model fit, times fit running time and saves the model.
        """
        t0 = time()
        self.model.fit(X, Y, **kwargs)
        t1 = time()
        lapse = t1 - t0
        self.train_time = lapse
        print(f"{self.name} took {lapse}s to train.")
        self.save()


def init_args():
    """Init command line args used for configuration."""

    parser = init_main_args()
    return parser.parse_args()

def parse_args(args):
    """Parse provided args for runtime configuration."""
    params = parse_main_args(args)
    params.update({'train': True})
    return params

def parse_args1(args):
    """Parse provided args for runtime configuration."""
    params = parse_main_args(args)
    params.update({'train': False})
    return params

#registry
_BINARY_MODELS = dict()

def register_bin(name):
    """Registers a new binary classification anomaly detection model."""

    def add_to_dict(func):
        _BINARY_MODELS[name] = func
        return func

    return add_to_dict

def get_binary_model(model):
    """Fetches the binary classification anomaly detection model."""
    return _BINARY_MODELS[model]

_MULTI_MODELS = dict()

def register_mul(name):
    """Registers a new multi-class anomaly classification model."""

    def add_to_dict(func):
        _MULTI_MODELS[name] = func
        return func

    return add_to_dict


def get_multi_model(model):
    """Fetches the multi-class anomaly classification model."""
    return _MULTI_MODELS[model]
class PUAdapterWrapper(BaseModel):
    def __init__(self, model, params):
        super().__init__(model, params)

    def save(self, **kwargs):
        pu_estimator_file = os.path.join(
            self.params['models_dir'],
            "pu_estimator.pkl"
            )
        pu_saver = {'estimator': self.model.estimator,
                    'c': self.model.c}
        with open(pu_estimator_file, 'wb') as pu_estimator_file:
            pickle.dump(pu_saver, pu_estimator_file)

    def load(self, **kwargs):
        pu_estimator_file = os.path.join(
            self.params['models_dir'],
            "pu_estimator.pkl"
            )
        with open(pu_estimator_file, 'rb') as pu_estimator_file:
            pu_saver = pickle.load(pu_estimator_file)
            estimator = pu_saver['estimator']
            pu_estimator = PUAdapter(estimator)
            pu_estimator.c = pu_saver['c']
            pu_estimator.estimator_fitted = True
            self.model = pu_estimator


@register_bin("pu_learning")
def instatiate_pu_adapter(params, **kwargs):
    """
        Returns a RF adapted to do PU Learning wrapped by the PUAdapterWrapper.
    """
    hparms = {
        'n_estimators': 10,
        'criterion': "entropy",
        'bootstrap': True,
        'n_jobs': -1,
    }
    hparms.update(kwargs)
    estimator = RandomForestClassifier(**hparms)
    wrapped_pu_estimator = PUAdapterWrapper(PUAdapter(estimator), params)
    return wrapped_pu_estimator


class RegularClassifierWrapper(BaseModel):
    def __init__(self, model, params):
        super().__init__(model, params)

    def save(self, **kwargs):
        regular_file = os.path.join(
            self.params['models_dir'],
            "regular.pkl"
            )
        with open(regular_file, 'wb') as regular_clf_file:
            pickle.dump(self.model, regular_clf_file)

    def load(self, **kwargs):
        regular_file = os.path.join(
            self.params['models_dir'],
            "regular.pkl"
            )
        with open(regular_file, 'rb') as regular_clf_file:
            regular_classifier = pickle.load(regular_clf_file)
            self.model = regular_classifier


@register_bin("regular")
def instatiate_regular_classifier(params, **kwargs):
    """
        Returns a RF wrapped by the PU Learning Adapter.
    """
    hparms = {
        'n_estimators': 10,
        'bootstrap': True,
        'n_jobs': -1,
    }
    hparms.update(kwargs)
    wrapped_regular = RegularClassifierWrapper(
        RandomForestClassifier(**hparms), params)
    return wrapped_regular

class SVMWrapper(BaseModel):
    def __init__(self, model, params):
        super().__init__(model, params)

    def save(self, **kwargs):
        multi_file = os.path.join(
            self.params['models_dir'],
            "multi.pkl"
            )
        with open(multi_file, 'wb') as multi_clf_file:
            pickle.dump(self.model, multi_clf_file)

    def load(self, **kwargs):
        multi_file = os.path.join(
            self.params['models_dir'],
            "multi.pkl"
            )
        with open(multi_file, 'rb') as multi_clf_file:
            multi_classifier = pickle.load(multi_clf_file)
            self.model = multi_classifier


@register_mul("svm")
def instatiate_svm(params, **kwargs):
    """
        Returns a RF wrapped by the PU Learning Adapter.
    """
    hparms = {
        'penalty': "l2",
        'dual': False,
        'tol': 1e-1,
    }
    hparms.update(kwargs)
    wrapped_svm = SVMWrapper(LinearSVC(**hparms), params)
    return wrapped_svm

_BB_REPORTS = dict()

#bb-registry
def register_bb(name):
    """Registers a new black box report or metric function."""

    def add_to_dict(func):
        _BB_REPORTS[name] = func
        return func

    return add_to_dict

def get_bb_report(model):
    """Fetches the black box report or metric function."""
    return _BB_REPORTS[model]

@register_bb('acc')
def model_accuracy(y, pred):
    return accuracy_score(y, pred)


@register_bb('confusion_matrix')
def report(y, pred):
    return confusion_matrix(y, pred)

@register_bb('macro')
def model_accuracy(y, pred):
    return f1_score(y, pred, average='macro')

@register_bb('micro')
def model_accuracy(y, pred):
    return f1_score(y, pred, average='micro')

@register_bb('multi_acc')
def model_accuracy(y, pred):
    return accuracy_score(y, pred)

def get_feature_names(params, vocabulary, add_length=True):
    feature_names = zip(vocabulary.keys(), vocabulary.values())
    feature_names = sorted(feature_names, key=lambda x: x[1])
    feature_names = [x[0] for x in feature_names]
    if 'length' in params['features']:
        feature_names.append('LENGTH')
    return np.array(feature_names)

#wb-registry
_WB_REPORTS = dict()
def register_wb(name):
    """Registers a new white box report or metric function."""

    def add_to_dict(func):
        _WB_REPORTS[name] = func
        return func

    return add_to_dict

@register_wb('top_k_svm')
def get_top_k_SVM_features(params, model, vocabulary, **kwargs):
    hparms = {
        'target_names': [],
        'top_features': 5,
    }
    hparms.update(kwargs)

    top_k_label = {}
    feature_names = get_feature_names(params, vocabulary)
    for i, label in enumerate(hparms['target_names']):
        if len(hparms['target_names']) < 3 and i == 1:
            break  # coef is unidemensional when there's only two labels
        coef = model.coef_[i]
        top_coefficients = np.argsort(coef)[-hparms['top_features']:]
        top_k_features = feature_names[top_coefficients]
        top_k_label[label] = list(reversed(top_k_features))
    return top_k_label


def get_wb_report(model):
    """Fetches the white box report or metric function."""
    return _WB_REPORTS[model]

# Compiling for optimization
re_sub_1 = re.compile(r"(:(?=\s))|((?<=\s):)")
re_sub_2 = re.compile(r"(\d+\.)+\d+")
re_sub_3 = re.compile(r"\d{2}:\d{2}:\d{2}")
re_sub_4 = re.compile(r"Mar|Apr|Dec|Jan|Feb|Nov|Oct|May|Jun|Jul|Aug|Sep")
re_sub_5 = re.compile(r":?(\w+:)+")
re_sub_7 = re.compile(r"[0-9]|\,")
re_sub_6 = re.compile(r"\.|\(|\)|\<|\>|\/|\-|\=|\[|\]")
p = re.compile(r"[^(A-Za-z)]")
def remove_parameters(msg):
    
    # Removing parameters with Regex
    msg = re.sub(re_sub_1, "", msg)
    msg = re.sub(re_sub_2, "", msg)
    msg = re.sub(re_sub_3, "", msg)
    msg = re.sub(re_sub_4, "", msg)
    msg = re.sub(re_sub_5, "", msg)
    msg = re.sub(re_sub_7, " ", msg)
    L = msg.split()
    # Filtering strings that have non-letter tokens
    #new_msg = [k for k in L if not p.search(k)]
   # msg = " ".join(new_msg)
    return msg

def remove_parameters_slower(msg):
    # Removing parameters with Regex
    msg = re.sub(r"(:(?=\s))|((?<=\s):)", "", msg)
    msg = re.sub(r"(\d+\.)+\d+", "", msg)
    msg = re.sub(r"\d{2}:\d{2}:\d{2}", "", msg)
    msg = re.sub(r"Mar|Apr|Dec|Jan|Feb|Nov|Oct|May|Jun|Jul|Aug|Sep", "", msg)
    msg = re.sub(r":?(\w+:)+", "", msg)
    msg = re.sub(r"\.|\(|\)|\<|\>|\/|\-|\=|\[|\]", " ", msg)
    L = msg.split()
    p = re.compile("[^(A-Za-z)]")
    # Filtering strings that have non-letter tokens
    new_msg = [k for k in L if not p.search(k)]
    msg = " ".join(new_msg)
    return msg

def get_ngrams(n, line):
    line = line.strip().split()
    cur_len = len(line)
    ngrams_list = []
    if cur_len == 0:
        # Token list is empty
        pass
    elif cur_len < n:
        # Token list fits in one ngram
        ngrams_list.append(" ".join(line))
    else:
        # Token list spans multiple ngrams
        loop_num = cur_len - n + 1
        for i in range(loop_num):
            cur_gram = " ".join(line[i: i + n])
            ngrams_list.append(cur_gram)
    return ngrams_list


def tokenize(line):
    return line.strip().split()


@print_step
def build_vocabulary(inputData):
    vocabulary = {}
    for line in inputData:
        token_list = tokenize(line)
        for token in token_list:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    return vocabulary


@print_step
def log_to_vector(inputData, vocabulary):
    result = []
    for line in inputData:
        temp = []
        token_list = tokenize(line)
        if token_list:
            for token in token_list:
                if token not in vocabulary:
                    continue
                else:
                    temp.append(vocabulary[token])
        result.append(temp)
    return np.array(result)


def setTrainDataForILF(x, y):
    x_res, indices = np.unique(x, return_index=True)
    y_res = y[indices]
    return x_res, y_res


def calculate_inv_freq(total, num):
    return np.log(float(total) / float(num + 0.01))


def get_max_line(inputVector):
    return len(max(inputVector, key=len))


def get_tf(inputVector):
    token_index_dict = defaultdict(set)
    # Counting the number of logs the word appears in
    for index, line in enumerate(inputVector):
        for token in line:
            token_index_dict[token].add(index)
    return token_index_dict


def get_lf(inputVector):
    token_index_ilf_dict = defaultdict(set)
    for line in inputVector:
        for location, token in enumerate(line):
            token_index_ilf_dict[token].add(location)
    return token_index_ilf_dict


def calculate_idf(token_index_dict, inputVector):
    idf_dict = {}
    total_log_num = len(inputVector)
    for token in token_index_dict:
        idf_dict[token] = calculate_inv_freq(total_log_num,
                                             len(token_index_dict[token]))
    return idf_dict


def calculate_ilf(token_index_dict, inputVector):
    ilf_dict = {}
    max_length = get_max_line(inputVector)
    # calculating ilf for each token
    for token in token_index_dict:
        ilf_dict[token] = calculate_inv_freq(max_length,
                                             len(token_index_dict[token]))
    return ilf_dict


def create_invf_vector(inputVector, invf_dict, vocabulary):
    tfinvf = []
    # Creating the idf/ilf vector for each log message
    for line in inputVector:
        cur_tfinvf = np.zeros(len(vocabulary))
        count_dict = Counter(line)
        for token_index in line:
            cur_tfinvf[token_index] = (
                float(count_dict[token_index]) * invf_dict[token_index]
            )
        tfinvf.append(cur_tfinvf)
    tfinvf = np.array(tfinvf)
    return tfinvf


def normalize_tfinvf(tfinvf):
    return 2.*(tfinvf - np.min(tfinvf))/np.ptp(tfinvf)-1


def calculate_tf_invf_train(
    inputVector, get_f=get_tf, calc_invf=calculate_idf
):
    token_index_dict = get_f(inputVector)
    invf_dict = calc_invf(token_index_dict, inputVector)
    return invf_dict

@print_step
def process_logs(input_source, output, process_line=None):
    with open(output, "w", encoding='latin-1') as f:
        # counting first to show progress with tqdm
        with open(input_source, 'r', encoding='latin-1') as IN:
            line_count = sum(1 for line in IN)
        with open(input_source, 'r', encoding='latin-1') as IN:
            with Pool() as pool:
                results = pool.imap(process_line, IN, chunksize=10000)
                f.writelines(tqdm(results, total=line_count))


@print_step
def load_logs(params, ignore_unlabeled=False):
    log_path = params['logs']
    unlabel_label = params['healthy_label']
    #print("unlabeled_label",unlabel_label)
    x_data = []
    y_data = []
    label_dict = {}
    target_names = []
    with open(log_path, 'r', encoding='latin-1') as IN:
        line_count = sum(1 for line in IN)
    with open(log_path, 'r', encoding='latin-1') as IN:
        for line in tqdm(IN, total=line_count):
            L = line.strip().split()
            label = L[0]
            if label not in label_dict:
                if ignore_unlabeled and label == unlabel_label:
                    continue
                if label == unlabel_label:
                    label_dict[label] = -1.0
                elif label not in label_dict:
                    label_dict[label] = len(label_dict)
                    target_names.append(label)
            x_data.append(" ".join(L[1:]))
            y_data.append(label_dict[label])
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    np.set_printoptions(threshold=sys.maxsize)  
    return x_data, y_data, target_names


_PREPROCESSORS = dict()

def register_pre(name):
    """Registers a new logs preprocessor function under the given name."""

    def add_to_dict(func):
        _PREPROCESSORS[name] = func
        return func

    return add_to_dict


def get_preprocessor(data_src):
    """Fetches the logs preprocessor function associated with the given raw logs"""
    print(_PREPROCESSORS.keys())
    return _PREPROCESSORS[data_src]

@register_pre("length")
def create_length_feature(params, input_vector, **kwargs):
    length = np.vectorize(len)
    length_feature = length(input_vector)
    length_feature = length_feature.reshape(-1, 1)
    return length_feature
_FEATURE_EXTRACTORS = dict()


def register_fea(name):
    """Registers a new log message feature extraction function under the
    given name."""

    def add_to_dict(func):
        _FEATURE_EXTRACTORS[name] = func
        return func

    return add_to_dict


def get_feature_extractor(feature):
    """Fetches the feature extraction function associated with the given
    raw logs"""
    return _FEATURE_EXTRACTORS[feature]

@register_fea("tfidf")
def create_tfidf_feature(params, train_vector, **kwargs):
    """
        Returns the tf-idf matrix of features.
    """
    if params['train']:
        invf_dict = calculate_tf_invf_train(
            train_vector,
            get_f=get_tf,
            calc_invf=calculate_idf
            )
        save_feature_dict(params, invf_dict, "tfidf")
    else:
        invf_dict = load_feature_dict(params, "tfidf")

    features = create_invf_vector(
        train_vector, invf_dict, kwargs['vocabulary'])
    return features

@register_fea("tfilf")
def create_tfilf_feature(params, train_vector, **kwargs):
    """
        Returns the tf-ilf matrix of features.
    """
    if params['train']:
        invf_dict = calculate_tf_invf_train(
            train_vector,
            get_f=get_lf,
            calc_invf=calculate_ilf
            )
        save_feature_dict(params, invf_dict, "tfilf")
    else:
        invf_dict = load_feature_dict(params, "tfilf")

    features = create_invf_vector(
        train_vector, invf_dict, kwargs['vocabulary'])
    return features

def create_tf_vector(input_vector, tf_dict, vocabulary):
    tf_vector = []
    # Creating the idf/ilf vector for each log message
    for line in input_vector:
        cur_tf_vector = np.zeros(len(vocabulary))
        for token_index in line:
            cur_tf_vector[token_index] = len(tf_dict[token_index])
        tf_vector.append(cur_tf_vector)

    tf_vector = np.array(tf_vector)
    return tf_vector


@register_fea("tf")
def create_term_count_feature(params, input_vector, **kwargs):
    """
        Returns an array of the counts of each word per log message.
    """
    if params['train']:
        tf_dict = get_tf(input_vector)
        save_feature_dict(params, tf_dict, "tf")
    else:
        tf_dict = load_feature_dict(params, "tf")

    tf_features =\
        create_tf_vector(input_vector, tf_dict, kwargs['vocabulary'])

    return tf_features
def load_feature_dict(params, name):
    dict_file = os.path.join(params['features_dir'], f"{name}.pkl")
    with open(dict_file, "rb") as fp:
        feat_dict = pickle.load(fp)
    return feat_dict


def save_feature_dict(params, feat_dict, name):
    dict_file = os.path.join(params['features_dir'], f"{name}.pkl")
    with open(dict_file, "wb") as fp:
        pickle.dump(feat_dict, fp)


def binary_train_gtruth(y):
    return np.where(y == 0.0, -1.0, 1.0)


def multi_features(x, y):
    anomalous = (y != -1)
    x_multi, y_multi = x[anomalous], y[anomalous]
    return x_multi, y_multi


@print_step
def get_features_vector(log_vector, vocabulary, params):
    feature_vectors = []
    for feature in params['features']:
        extract_feature = get_feature_extractor(feature)
        feature_vector = extract_feature(
            params, log_vector, vocabulary=vocabulary)
        feature_vectors.append(feature_vector)
    X = np.hstack(feature_vectors)
    return X


@print_step
def extract_features(x, params):
    # Build Vocabulary
    if params['train']:
        vocabulary = build_vocabulary(x)
        save_feature_dict(params, vocabulary, "vocab")
    else:
        vocabulary = load_feature_dict(params, "vocab")
    # Feature Engineering
    x_vector = log_to_vector(x, vocabulary)
    x_features = get_features_vector(x_vector, vocabulary, params)
    return x_features, vocabulary

def process_line(line):
    label = line[0].strip()
    msg = ' '.join(line[1].strip().split()[1:])
    msg = remove_parameters(msg)
    if msg:
        msg = ' '.join((label, msg))
        msg = ''.join((msg, '\n'))
        return msg
    return ''


def process_open_source(input_source, output):
    with open(output, "w", encoding='latin-1') as f:
        gtruth = "./LogClass/data/intern/APP1/preprocessed_logs_weight.txt"
        rawlog = "./LogClass/data/intern/APP1/preprocessed_logs_word.txt"
        with open(gtruth, 'r', encoding='latin-1') as IN:
            line_count = sum(1 for line in IN)
        with open(gtruth, 'r', encoding='latin-1') as in_gtruth:
            with open(rawlog, 'r', encoding='latin-1') as in_log:
                IN = zip(in_gtruth, in_log)
                with Pool() as pool:
                    results = pool.imap(process_line, IN, chunksize=10000)
                    f.writelines(tqdm(results, total=line_count))


open_source_datasets = [
    'open_Apache',
    'open_bgl',
    'open_hadoop',
    'open_hdfs',
    'open_hpc',
    'open_proxifier',
    'open_zookeeper',
]
for dataset in open_source_datasets:
    @register_pre(dataset)
    def preprocess_dataset(params):
        """
        Runs open source logs preprocessing executor.
        """
        input_source = os.path.join(
            params['raw_logs'],
            dataset.split('_')[-1]
        )
        output = params['logs']
        params['healthy_label'] = 'NA'
        process_open_source(input_source, output)



def inference(params, x_data, y_data, target_names):
    # Inference
    # Feature engineering
    x_test, _ = extract_features(x_data, params)
    file_fea = open("extract_features.txt","w")
    for i in range(len(x_test)):
        file_fea.write(str(x_test[i]))
        file_fea.write("\n")
    file_fea.close()
    # Binary training features
    y_test = binary_train_gtruth(y_data)
    # Binary PU estimator with RF
    # Load Trained PU Estimator
    binary_clf_getter =\
        get_binary_model(
            params['binary_classifier'])
    binary_clf = binary_clf_getter(params)
    data = binary_clf.load()
    #print(data)
    #0
    # print(binary_clf.head())
    # Anomaly detectionf
    y_pred_pu = binary_clf.predict(x_test)
    print(len(y_pred_pu))
    # for i in range(len(y_pred_pu)):
    #     if(y_pred_pu[i]==-1):
    #         print(i)
    get_accuracy = get_bb_report('acc')
    get_conf = get_bb_report('confusion_matrix')
    binary_conf = get_conf(y_test, y_pred_pu)
    print("Confusion_matrix: \n",binary_conf)
    binary_acc = get_accuracy(y_test, y_pred_pu)
    print(binary_acc)
    tp_and_fn = binary_conf.sum(1)
    tp_and_fp = binary_conf.sum(0)
    tp = binary_conf.diagonal()
    precision = tp / tp_and_fp
    recall = tp / tp_and_fn
    print("precision = ",precision,"\nrecall = ",recall,"\n")
    for report in params['report']:
        try:
            get_bb_report1 = get_bb_report(report)
            result = get_bb_report1(y_test, y_pred_pu)
        except Exception:
            pass
        else:
            print(f'Binary classification {report} report:')
            print(result)


def train(params, x_data, y_data, target_names):
    # KFold Cross Validation
    kfold = StratifiedKFold(n_splits=3).split(x_data, y_data)
    best_pu_fs = 0.
    for train_index, test_index in tqdm(kfold):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        x_train, _ = extract_features(x_train, params)
        #print("x_Train\n",x_train)
        with TestingParameters(params):
            x_test, _ = extract_features(x_test, params)
        # Binary training features
        y_test_pu = binary_train_gtruth(y_test)
        y_train_pu = binary_train_gtruth(y_train)
        # Binary PULearning with RF
        binary_clf_getter =\
            get_binary_model(
                params['binary_classifier'])
        binary_clf = binary_clf_getter(params)
        binary_clf.fit(x_train, y_train_pu)
        y_pred_pu = binary_clf.predict(x_test)

        get_accuracy = get_bb_report('acc')
        get_conf = get_bb_report('confusion_matrix')
        binary_conf = get_conf(y_test_pu, y_pred_pu)
        binary_acc = get_accuracy(y_test_pu, y_pred_pu)
        print("Confusion_matrix: \n",binary_conf)
        tp_and_fn = binary_conf.sum(1)
        tp_and_fp = binary_conf.sum(0)
        tp = binary_conf.diagonal()
        precision = tp / tp_and_fp
        recall = tp / tp_and_fn
        print("precision = ",precision,"\nrecall = ",recall,"\n")
        better_results = binary_acc > best_pu_fs
        if better_results:
            if binary_acc > best_pu_fs:
                best_pu_fs = binary_acc
            save_params(params)
            binary_clf.save()
            print("acc",binary_acc)

        for report in params['report']:
            try:
                get_bb_report1 = get_bb_report(report)
                result = get_bb_report1(y_test_pu, y_pred_pu)
            except Exception:
                pass
            else:
                print(f'Binary classification {report} report:')
                print(result)


def main():
    #TRAIN
    # print("\t\t\t\t\t\tTRAIN")
    #  # Init params: train = 'true'
    # params = parse_args(init_args())
    # #storing experiment ID of TRAIN
    # file_handling(params)
    # # Filter params from raw logs
    # if "raw_logs" in params:
    #     preprocess = get_preprocessor(params['logs_type'])
    #     preprocess(params)
    # # Load filtered params from file
    # print('Loading logs')
    # x_data, y_data, target_names = load_logs(params)
    # print_params(params)
    # train(params, x_data, y_data, target_names)
    
    # INFERENCE
    print("\t\t\t\t\t\tINFERENCE")
    # init params: to make train = 'false'
    params = parse_args1(init_args())
    # Filter params from raw logs
    if "raw_logs" in params:
        preprocess = get_preprocessor(params['logs_type'])
        preprocess(params)
    # Load filtered params from file
    x_data, y_data, target_names = load_logs(params)
    inference(params, x_data, y_data, target_names)


if __name__ == "__main__":
    main()
