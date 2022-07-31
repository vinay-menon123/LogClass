from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import os
import shutil
import json
import sys
import numpy as np
import pickle
import argparse
from uuid import uuid4



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
        "--swap",
        action="store_true",
        default=False,
        help="Swap testing/training data in kfold cross validation.",
    )

    return parser

def parse_main_args(args):
    """Parse provided args for runtime configuration."""
    params = {
        "report": args.report,
        "train": args.train,
        "force": args.force,
        "base_dir": args.base_dir[0],
        "logs_type": args.logs_type[0],
        "healthy_label": args.healthy_label[0],
        "features": args.features,
        "binary_classifier": args.binary_classifier[0],
        "multi_classifier": args.multi_classifier[0],
        "swap": args.swap,
    }
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
    if args.id:
        params['id'] = args.id[0]
    else:
        params['id'] = str(uuid4().time_low)
    print(f"\nExperiment ID: {params['id']}")
    # Creating experiments results folder with the format
    # {experiment_module_name}_{logs_type}_{id}
    experiment_name = os.path.basename(sys.argv[0]).split('.')[0]
    params['id_dir'] = os.path.join(
            params['base_dir'],
            '_'.join((
                experiment_name, params['logs_type']
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

_BB_REPORTS = dict()
def register(name):
    """Registers a new black box report or metric function."""

    def add_to_dict(func):
        _BB_REPORTS[name] = func
        return func

    return add_to_dict


def get_bb_report(model):
    """Fetches the black box report or metric function."""
    return _BB_REPORTS[model]

_FEATURE_EXTRACTORS = dict()
def register(name):
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

_BINARY_MODELS = dict()
def register(name):
    """Registers a new binary classification anomaly detection model."""

    def add_to_dict(func):
        _BINARY_MODELS[name] = func
        return func

    return add_to_dict


def get_binary_model(model):
    """Fetches the binary classification anomaly detection model."""
    return _BINARY_MODELS[model]


def get_features_vector(log_vector, vocabulary, params):
    feature_vectors = []
    for feature in params['features']:
        extract_feature = get_feature_extractor(feature)
        feature_vector = extract_feature(
            params, log_vector, vocabulary=vocabulary)
        feature_vectors.append(feature_vector)
    X = np.hstack(feature_vectors)
    return X

def load_feature_dict(params, name):
    dict_file = os.path.join(params['features_dir'], f"{name}.pkl")
    with open(dict_file, "rb") as fp:
        feat_dict = pickle.load(fp)
    return feat_dict

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

def save_feature_dict(params, feat_dict, name):
    dict_file = os.path.join(params['features_dir'], f"{name}.pkl")
    with open(dict_file, "wb") as fp:
        pickle.dump(feat_dict, fp)

def tokenize(line):
    return line.strip().split()

def build_vocabulary(inputData):
    vocabulary = {}
    for line in inputData:
        token_list = tokenize(line)
        for token in token_list:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    return vocabulary

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

def binary_train_gtruth(y):
    return np.where(y == 0.0, 1.0, -1.0)

def load_logs(params, ignore_unlabeled=False):
    log_path = params['logs']
    unlabel_label = params['healthy_label']
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
    return x_data, y_data, target_names

_PREPROCESSORS = dict()
def register(name):
    """Registers a new logs preprocessor function under the given name."""

    def add_to_dict(func):
        _PREPROCESSORS[name] = func
        return func

    return add_to_dict


def get_preprocessor(data_src):
    """Fetches the logs preprocessor function associated with the given raw logs"""
    return _PREPROCESSORS[data_src]


def print_params(params):
    print("{:-^80}".format("params"))
    print("Beginning experiment using the following configuration:\n")
    for param, value in params.items():
        print("\t{:>13}: {}".format(param, value))
    print()
    print("-" * 80)

class TestingParameters():
    def __init__(self, params):
        self.params = params
        self.original_state = params['train']

    def __enter__(self):
        self.params['train'] = False

    def __exit__(self, exc_type, exc_value, traceback):
        self.params['train'] = self.original_state

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

def save_params(params):
    params_file = os.path.join(
        params['id_dir'], f"best_params.json")
    with open(params_file, "w") as fp:
        json.dump(params, fp)

def init_args():
    """Init command line args used for configuration."""

    parser = init_main_args()
    return parser.parse_args()

def parse_args(args):
    """Parse provided args for runtime configuration."""
    params = parse_main_args(args)
    params.update({'train': True})
    return params

def parse_args1(args, id):
    """Parse provided args for runtime configuration."""
    params = parse_main_args(args)
    params.update({'train': False})
    return params

def train(params, x_data, y_data, target_names):
    # KFold Cross Validation
    kfold = StratifiedKFold(n_splits=3).split(x_data, y_data)
    best_pu_fs = 0.
    for train_index, test_index in tqdm(kfold):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        x_train, _ = extract_features(x_train, params)
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
        binary_acc = get_accuracy(y_test_pu, y_pred_pu)
        better_results = binary_acc > best_pu_fs
        if better_results:
            if binary_acc > best_pu_fs:
                best_pu_fs = binary_acc
            save_params(params)
            binary_clf.save()
            print(binary_acc)

        for report in params['report']:
            try:
                get_bb_report = get_bb_report(report)
                result = get_bb_report(y_test_pu, y_pred_pu)
            except Exception:
                pass
            else:
                print(f'Binary classification {report} report:')
                print(result)

def inference(params, x_data, y_data, target_names):
    # Inference
    # Feature engineering
    x_test, _ = extract_features(x_data, params)
    # Binary training features
    y_test = binary_train_gtruth(y_data)
    # Binary PU estimator with RF
    # Load Trained PU Estimator
    binary_clf_getter =\
        get_binary_model(
            params['binary_classifier'])
    binary_clf = binary_clf_getter(params)
    binary_clf.load()
    # Anomaly detection
    y_pred_pu = binary_clf.predict(x_test)
    get_accuracy = get_bb_report('acc')
    binary_acc = get_accuracy(y_test, y_pred_pu)

    print(binary_acc)
    for report in params['report']:
        try:
            get_bb_report = get_bb_report(report)
            result = get_bb_report(y_test, y_pred_pu)
        except Exception:
            pass
        else:
            print(f'Binary classification {report} report:')
            print(result)


def main():
    print("\t\t\t\t\t\tTRAIN")
     # Init params: train = 'true'
    params = parse_args(init_args())
    #storing experiment ID of TRAIN
    expid = params['id']
    file_handling(params)
    # Filter params from raw logs
    if "raw_logs" in params:
        preprocess = get_preprocessor(params['logs_type'])
        preprocess(params)
    # Load filtered params from file
    print('Loading logs')
    x_data, y_data, target_names = load_logs(params)
    print_params(params)
    train(params, x_data, y_data, target_names)

    # print("\t\t\t\t\t\tINFERENCE")
    # # init params: to make train = 'false'
    # params = parse_args1(init_args(), id)
    # # Filter params from raw logs
    # if "raw_logs" in params:
    #     preprocess = get_preprocessor(params['logs_type'])
    #     preprocess(params)
    # # Load filtered params from file
    # x_data, y_data, target_names = load_logs(params)
    # inference(params, x_data, y_data, target_names)


if __name__ == "__main__":
    main()

