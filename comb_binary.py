from .utils import (
    load_params,
    save_params,
    file_handling,
    TestingParameters,
    print_params,
)
from .preprocess import registry as preprocess_registry
from .preprocess.utils import load_logs
from .feature_engineering.utils import (
    binary_train_gtruth,
    extract_features,
)
from .models import binary_registry as binary_classifier_registry
from .reporting import bb_registry as black_box_report_registry
from .init_params import init_main_args, parse_main_args
from .modified_params import parse_main_args_mod,init_main_args_mod
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

def init_args():
    """Init command line args used for configuration."""

    parser = init_main_args()
    return parser.parse_args()
def init_args_mod():
    """Init command line args used for configuration."""

    parser = init_main_args_mod()
    return parser.parse_args()

def parse_args(args):
    """Parse provided args for runtime configuration."""
    params = parse_main_args(args)
    params.update({'train': True})
    return params
    
def parse_args_mod(args):
    """Parse provided args for runtime configuration."""
    #args.id = id_val
    params = parse_main_args_mod(args)
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
            binary_classifier_registry.get_binary_model(
                params['binary_classifier'])
        binary_clf = binary_clf_getter(params)
        binary_clf.fit(x_train, y_train_pu)
        y_pred_pu = binary_clf.predict(x_test)
        get_accuracy = black_box_report_registry.get_bb_report('acc')
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
                get_bb_report = black_box_report_registry.get_bb_report(report)
                result = get_bb_report(y_test_pu, y_pred_pu)
            except Exception:
                pass
            else:
                print(f'Binary classification {report} report:')
                print(result)
    id_value = params['id']
    return id_value

def inference(params, x_data, y_data, target_names,id_val):
    # Inference
    # Feature engineering
    params.update({'id': id_val})
    #params['id'] = id_val
    print("ID value",params['id'])
    x_test, _ = extract_features(x_data, params)
    # Binary training features
    y_test = binary_train_gtruth(y_data)
    # Binary PU estimator with RF
    # Load Trained PU Estimator
    binary_clf_getter =\
        binary_classifier_registry.get_binary_model(
            params['binary_classifier'])
    binary_clf = binary_clf_getter(params)
    binary_clf.load()
    # Anomaly detection
    y_pred_pu = binary_clf.predict(x_test)
    get_accuracy = black_box_report_registry.get_bb_report('acc')
    binary_acc = get_accuracy(y_test, y_pred_pu)

    print(binary_acc)
    for report in params['report']:
        try:
            get_bb_report = black_box_report_registry.get_bb_report(report)
            result = get_bb_report(y_test, y_pred_pu)
        except Exception:
            pass
        else:
            print(f'Binary classification {report} report:')
            print(result)


def main():
    print("\t\t\t\t\t\tTRAIN")
     # Init params
    params = parse_args(init_args())
    file_handling(params)
    # Filter params from raw logs
    if "raw_logs" in params:
        preprocess = preprocess_registry.get_preprocessor(params['logs_type'])
        preprocess(params)
    # Load filtered params from file
    print('Loading logs')
    x_data, y_data, target_names = load_logs(params)
    print_params(params)
    id_val = train(params, x_data, y_data, target_names)

    print("\t\t\t\t\t\tINFERENCE")
    # load_params(params)
    # print_params(params)
    # file_handling(params)
    # # Filter params from raw logs
    params = parse_args_mod(init_args_mod())
    if "raw_logs" in params:
        preprocess = preprocess_registry.get_preprocessor(params['logs_type'])
        preprocess(params)
    # # Load filtered params from file
    # print('Loading logs')
    x_data, y_data, target_names = load_logs(params)
    inference(params, x_data, y_data, target_names,id_val)


if __name__ == "__main__":
    main()
