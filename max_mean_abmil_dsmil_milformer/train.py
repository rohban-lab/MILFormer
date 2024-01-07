#mostly ours and very few parts from https://github.com/binli123/dsmil-wsi
import pickle
from typing import Tuple, Optional, List
import argparse, os, copy, sys, time, itertools, json, re

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import torch
from torch.nn.modules.loss import _Loss
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from lightly.utils.scheduler import CosineWarmupScheduler
import wandb

from utils import (
    WEIGHT_INITS, OPTIMIZERS,
    pretty_print, print_table, replace_key_names, delete_files_for_epoch,
    to_wandb_format, NumpyFloatValuesEncoder,
    load_data, load_mil_data,
    dropout_patches, multi_label_roc, compute_pos_weight
)

import abmil
import dsmil
import milformer

print('Imports Finished.')

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise Exception("WE ONLY WANT GPU")

EMBEDDINGS_PATH = 'embeddings/'
SAVE_PATH = 'runs/'
ROC_PATH = 'roc/'
HISTOPATHOLOGY_DATASETS = ['camelyon16', 'tcga']
MIL_DATASETS = ['musk1', 'musk2', 'elephant', 'fox', 'tiger']


def get_args_parser():
    parser = argparse.ArgumentParser(description='Train MIL Models on patch features learned by the SSL method')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--eta_min', default=5e-06)
    parser.add_argument('--dataset', default='camelyon16', type=str, help='Dataset folder name')
    parser.add_argument('--embedding', default='SimCLR', type=str, help='Embeddings to ba used for feature computation')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True,
                        help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--weight_init', default='xavier_normal',
                        choices=['xavier_normal', 'xavier_uniform', 'trunc_normal'],
                        type=str, help='weight initialization')
    parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'adamw', 'sgd'], help='optimizer')
    parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosinewarmup', 'cosine'],
                        help='scheduler')
    parser.add_argument('--num_processes', default=8, type=int,
                        help='number of processes for multiprocessing of data loading')

    parser.add_argument('--wandb_run',
                        help='Name for the wandb run. The model logs will be saved at `run/dataset/{wandb_run}_run_number/`')
    parser.add_argument('--use_mp', default=1, choices=[0, 1], type=int,
                        help='use multiprocessing for dataloading or not')
    parser.add_argument('--arch', default='dsmil', type=str, help='architecture')

    # For MIL datasets (Musk1, Musk2, Fox, Tiger, Elephant)
    parser.add_argument('--cv_num_folds', default=10, type=int, help='Number of cross validation fold [10]')
    parser.add_argument('--cv_current_fold', default=0, type=int, help='Current fold for cross validation')
    parser.add_argument('--cv_valid_ratio', default=0.2, type=float, help='Current fold for cross validation')

    # For LambdaTrainer (and its subclasses)
    parser.add_argument('--soft_average', default=0, choices=[0, 1], type=int)
    parser.add_argument('--lambda_lr_multiplier', default=0.1, type=float,
                        help='intial lr multiplied by this number for lambda')

    # For MILFormer
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--k', default=200, type=int, help='top k')

    # Only for MILFormer
    parser.add_argument('--random_patch_share', default=0.0, type=float, help='dropout in encoder')
    parser.add_argument('--mlp_multiplier', default=4, type=int, help='inverted mlp anti-bottbleneck')
    parser.add_argument('--encoder_dropout', default=0.0, type=float, help='dropout in encoder')

    # For wandb sweep
    parser.add_argument(
        '--seed', default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], type=int,
        help='This doesnt do anything, Each combination of the sweep should be run 5 times'
    )

    # For ROC curve
    parser.add_argument(
        '--roc_run_name', type=str, help="Name of the run for which we're saving predictions and labels."
    )
    parser.add_argument(
        '--roc_run_epoch', type=int, help="Epoch number of the run for which we're saving predictions and labels."
    )
    parser.add_argument(
        '--roc_data_split', default='test', type=str, choices=['train', 'valid', 'test'],
        help="Data Split for which we're saving predictions and labels"
    )

    return parser


class Trainer:
    def __init__(self, args):
        self.args = args
        self.milnet = self._get_milnet()
        self._load_init_weights()
        self.__is_criterion_set = False
        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

    def _get_milnet(self) -> nn.Module:
        raise NotImplementedError

    def _get_criterion(self) -> Optional[_Loss]:
        # For MIL datasets, For all models (ours and DSMIL) (not ABMIL), criterion should be weighted BCE,
        # where weights are determined by train split labels.
        self.__is_criterion_set = not (
                self.args.dataset in MIL_DATASETS and
                self.args.arch not in ['abmil', 'abmil-gated', 'abmil-paper', 'abmil-gated-paper']
        )
        return nn.BCEWithLogitsLoss()

    def _get_optimizer(self) -> optim.Optimizer:
        try:
            optimizer_cls = OPTIMIZERS[self.args.optimizer]
        except KeyError:
            raise Exception(f'Optimizer not found. Given: {self.args.optimizer}, Have: {OPTIMIZERS.keys()}')

        print(
            f'Optimizer {self.args.optimizer} with lr={self.args.lr}, betas={(0.5, 0.9)}, wd={self.args.weight_decay}'
        )
        return optimizer_cls(
            params=self.milnet.parameters(),
            lr=self.args.lr,
            betas=(0.5, 0.9),
            weight_decay=self.args.weight_decay
        )

    def _get_scheduler(self):
        if self.args.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=self.args.num_epochs,
                eta_min=self.args.eta_min
            )
        elif self.args.scheduler == 'cosinewarmup':
            return CosineWarmupScheduler(
                optimizer=self.optimizer,
                warmup_epochs=int(self.args.num_epochs / 20),
                max_epochs=self.args.num_epochs
            )
        else:
            print(f'Scheduler set to None')
            return None

    def _load_init_weights(self):
        try:
            weight_init = WEIGHT_INITS[self.args.weight_init]
            self.milnet.apply(weight_init)
        except KeyError:
            if self.args.weight_init is not None:
                raise Exception(f'Weight init not found. Given: {self.args.weight_init}, Have: {WEIGHT_INITS.keys()} ')

    @staticmethod
    def _should_calc_feats_metrics(data):
        """
        TCGA dataset doesn't have patch-level labels. Therefore, we can't calculate feat metrics for it.
        Official DSMIL-WSI features do not have patch-lebel labels either.
        """
        return data[2] is not None

    def train(self, data):
        self.milnet.train()
        if data[2] is not None:
            data = shuffle(data[0], data[1], data[2], data[3])
        else:
            data = shuffle(data[0], data[1])
            data = data[0], data[1], None, None
        all_labels, all_feats, all_feats_labels, all_positions = data
        num_bags = len(all_labels)
        Tensor = torch.cuda.FloatTensor

        total_loss = 0
        labels = all_labels
        predictions = []
        feat_labels = all_feats_labels
        feat_predictions = []

        if not self.__is_criterion_set:
            pos_weight = torch.tensor(compute_pos_weight(labels), device='cuda')
            self.criterion = nn.BCEWithLogitsLoss(pos_weight)
            self.__is_criterion_set = True

        for i in range(num_bags):
            self._before_run_model_in_training_mode()

            bag_label, bag_feats = labels[i], all_feats[i]
            bag_feats = dropout_patches(bag_feats, self.args.dropout_patch)
            bag_label = Variable(Tensor(np.array([bag_label])).cuda())  # .unsqueeze(dim=0)
            bag_feats = Variable(Tensor(np.array([bag_feats])).cuda())  # .unsqueeze(dim=0)
            # bag_feats = bag_feats.view(-1, self.args.feats_size) # milformer
            bag_prediction, loss, attentions = self._run_model(bag_feats, bag_label)
            loss.backward()

            self._after_run_model_in_training_mode()

            total_loss = total_loss + loss.item()
            step_train_metrics = {'step_train_bag_loss': loss.item()}
            wandb.log(step_train_metrics)
            sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, num_bags, loss.item()))
            with torch.no_grad():
                predictions.extend([bag_prediction])
                if self._should_calc_feats_metrics(data):
                    feat_predictions.extend(attentions.cpu().numpy().squeeze())

        labels = np.array(labels)
        predictions = np.array(predictions)
        accuracy, auc_scores, _, ap_scores, f1_scores = self._calc_metrics(labels, predictions)

        feats_accuracy, feats_auc_scores, feats_ap_scores, feats_f1_scores = None, None, None, None
        if self._should_calc_feats_metrics(data):
            feat_labels = list(itertools.chain(*feat_labels))  # convert a list of lists to a flat list
            feat_labels = np.array(feat_labels)
            feat_predictions = np.array(feat_predictions)
            feats_accuracy, feats_auc_scores, _, feats_ap_scores, feats_f1_scores = self._calc_feats_metrics(
                feat_labels, feat_predictions
            )

        res = {
            'epoch_train_loss': total_loss / num_bags,
            'epoch_train_accuracy': accuracy,
            'epoch_train_aucs': auc_scores,
            'epoch_train_aps': ap_scores,
            'epoch_train_f1s': f1_scores,
            'epoch_train_feat_accuracy': feats_accuracy,
            'epoch_train_feat_aucs': feats_auc_scores,
            'epoch_train_feat_aps': feats_ap_scores,
            'epoch_train_feat_f1s': feats_f1_scores,
        }
        return res

    def valid(self, data, predefined_thresholds_optimal=None, predefined_feats_thresholds_optimal=None):
        self.milnet.eval()
        if data[2] is not None:
            data = shuffle(data[0], data[1], data[2], data[3])
        else:
            data = shuffle(data[0], data[1])
            data = data[0], data[1], None, None
        all_labels, all_feats, all_feats_labels, all_positions = data
        num_bags = len(all_labels)
        Tensor = torch.cuda.FloatTensor

        total_loss = 0
        labels = all_labels
        predictions = []
        feat_labels = all_feats_labels  # +
        feat_predictions = []  # +

        with torch.no_grad():
            for i in range(num_bags):
                bag_label, bag_feats = labels[i], all_feats[i]
                bag_label = Variable(Tensor(np.array([bag_label])).cuda())
                bag_feats = Variable(Tensor(np.array([bag_feats])).cuda())
                # bag_feats = bag_feats.view(-1, self.args.feats_size) # milformer

                bag_prediction, loss, attentions = self._run_model(bag_feats, bag_label)

                total_loss = total_loss + loss.item()
                step_validation_metrics = {
                    'step_valid_bag_loss': loss.item()
                }
                wandb.log(step_validation_metrics)
                sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, num_bags, loss.item()))
                predictions.extend([bag_prediction])
                if self._should_calc_feats_metrics(data):
                    feat_predictions.extend(attentions.cpu().numpy().squeeze())  # +

        accuracy, auc_scores, thresholds_optimal, ap_scores, f1_scores = self._calc_metrics(
            labels, predictions, predefined_thresholds_optimal
        )
        if self.args.for_roc_curve:
            print(f'\nPredictions: {predictions}')
            print(f'Labels: {labels}')
            roc_base_dir = os.path.join(ROC_PATH, self.args.roc_run_name)
            os.makedirs(roc_base_dir, exist_ok=True)
            labels_predictions_f_path = os.path.join(roc_base_dir, f'{self.args.roc_run_epoch}.npz')
            np.savez(labels_predictions_f_path, labels=labels, predictions=predictions, )
            print(f'\n\nSaved at {labels_predictions_f_path}')

        feats_accuracy, feats_auc_scores, feats_thresholds_optimal, feats_ap_scores, feats_f1_scores = None, None, None, None, None
        if self._should_calc_feats_metrics(data):
            feat_labels = list(itertools.chain(*feat_labels))  # convert a list of lists to a flat list
            feat_labels = np.array(feat_labels)
            feat_predictions = np.array(feat_predictions)
            feats_accuracy, feats_auc_scores, feats_thresholds_optimal, feats_ap_scores, feats_f1_scores = \
                self._calc_feats_metrics(feat_labels, feat_predictions, predefined_feats_thresholds_optimal)

        res = {
            'epoch_valid_loss': total_loss / num_bags,
            'epoch_valid_accuracy': accuracy,
            'epoch_valid_aucs': auc_scores,
            'epoch_valid_aps': ap_scores,
            'epoch_valid_f1s': f1_scores,
            'epoch_valid_thresholds_optimal': thresholds_optimal,
            'epoch_valid_feat_accuracy': feats_accuracy,
            'epoch_valid_feat_aucs': feats_auc_scores,
            'epoch_valid_feats_thresholds_optimal': feats_thresholds_optimal,
            'epoch_valid_feat_aps': feats_ap_scores,
            'epoch_valid_feat_f1s': feats_f1_scores,
        }
        return res

    def test(self, data, thresholds_optimal, feats_thresholds_optimal):
        res = self.valid(data, thresholds_optimal, feats_thresholds_optimal)
        res = replace_key_names(d=res, old_term='valid', new_term='test')
        return res

    def _before_run_model_in_training_mode(self):
        self.optimizer.zero_grad()

    def _run_model(self, bag_feats, bag_label) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _after_run_model_in_training_mode(self):
        self.optimizer.step()

    def _calc_metrics(self, labels, predictions, predefined_thresholds_optimal=None):
        assert len(labels) == len(predictions), \
            f"Number of predictions ({len(predictions)}) and labels ({len(labels)}) do not match"

        num_bags = len(labels)
        labels = np.array(labels)
        predictions = np.array(predictions)
        auc_scores, _, thresholds_optimal, ap_scores, f1_scores = multi_label_roc(labels, predictions,
                                                                                  self.args.num_classes)

        if predefined_thresholds_optimal is not None:
            thresholds_optimal = predefined_thresholds_optimal

        if self.args.arch in ['abmil-paper', 'abmil-gated-paper']:
            thresholds_optimal = [0.5]

        if self.args.num_classes == 1:
            class_prediction_bag = copy.deepcopy(predictions)
            class_prediction_bag[predictions >= thresholds_optimal[0]] = 1
            class_prediction_bag[predictions < thresholds_optimal[0]] = 0
            predictions = class_prediction_bag
            labels = np.squeeze(labels)
        else:
            for i in range(self.args.num_classes):
                class_prediction_bag = copy.deepcopy(predictions[:, i])
                class_prediction_bag[predictions[:, i] >= thresholds_optimal[i]] = 1
                class_prediction_bag[predictions[:, i] < thresholds_optimal[i]] = 0
                predictions[:, i] = class_prediction_bag

        bag_score = 0
        for i in range(num_bags):
            bag_score += np.array_equal(labels[i], predictions[i])
        accuracy = bag_score / num_bags

        return accuracy, auc_scores, thresholds_optimal, ap_scores, f1_scores

    def _calc_feats_metrics(self, feats_labels, feats_predictions, predefined_thresholds_optimal=None):
        auc_scores, _, thresholds_optimal, ap_scores, f1_scores = multi_label_roc(
            feats_labels, feats_predictions, self.args.num_classes, for_feats=True
        )

        if predefined_thresholds_optimal is not None:
            thresholds_optimal = predefined_thresholds_optimal

        accuracy = accuracy_score(
            feats_labels,
            (feats_predictions >= thresholds_optimal[0]).astype(int)
        )

        return accuracy, auc_scores, thresholds_optimal, ap_scores, f1_scores


class Runner:
    def __init__(self, args, trainer: Trainer):
        self.args = args
        self.trainer = trainer
        self._set_dirs()

        if self.args.dataset in HISTOPATHOLOGY_DATASETS:
            if self.args.embedding == 'official':
                self.train_data, self.valid_data, self.test_data = self._get_official_data()
            else:
                self.train_data, self.valid_data, self.test_data = self._get_data()
        elif self.args.dataset in MIL_DATASETS:
            self.train_data, self.valid_data, self.test_data = load_mil_data(args)

        print(
            f'Num Bags'
            f' (Train: {len(self.train_data[0])})'
            f' (Valid: {len(self.valid_data[0])})'
            f' (Test: {len(self.test_data[0])})'
        )

    def _set_dirs(self):
        self.save_path = os.path.join(SAVE_PATH, self.args.dataset, wandb.run.name)
        os.makedirs(self.save_path, exist_ok=True)

    def _get_data(self):
        """
        bag_df:         [column_0]                  [column_1]
                        path_to_bag_feats_csv       label
        """
        path_prefix = os.path.join(EMBEDDINGS_PATH, self.args.dataset, self.args.embedding)

        bags_csv = os.path.join(path_prefix, self.args.dataset + '.csv')
        bags_df = pd.read_csv(bags_csv)
        train_df, valid_df, test_df = self._get_dataframe_splits_by_folder(bags_df, path_prefix)

        train_df = shuffle(train_df).reset_index(drop=True)
        valid_df = shuffle(valid_df).reset_index(drop=True)
        test_df = shuffle(test_df).reset_index(drop=True)

        print(f'Num Bags (Train: {len(train_df)}) (Valid: {len(valid_df)}) (Test: {len(test_df)})')

        train_data = self._load_split_data(train_df, 'train')
        valid_data = self._load_split_data(valid_df, 'valid')
        test_data = self._load_split_data(test_df, 'test')
        return train_data, valid_data, test_data

    def _get_official_data(self):
        bags_csv = os.path.join(EMBEDDINGS_PATH, self.args.dataset, 'official', f'{self.args.dataset.capitalize()}.csv')
        bags_df = pd.read_csv(bags_csv)
        train_df, valid_df, test_df = self._get_dataframe_splits_by_args(bags_df)

        train_df = shuffle(train_df).reset_index(drop=True)
        valid_df = shuffle(valid_df).reset_index(drop=True)
        test_df = shuffle(test_df).reset_index(drop=True)

        train_data = self._load_split_data(train_df, 'train')
        valid_data = self._load_split_data(valid_df, 'valid')
        test_data = self._load_split_data(test_df, 'test')

        return train_data, valid_data, test_data

    def _get_dataframe_splits_by_folder(self, bags_df, path_prefix):
        split_names = ['train', 'valid', 'test']
        dataframe_splits = (
            bags_df[
                bags_df['0'].str.startswith(f'{path_prefix}/{split_name}')
            ] for split_name in split_names
        )
        return dataframe_splits

    def _get_dataframe_splits_by_args(self, bags_df):
        train_df = bags_df.iloc[0:int(len(bags_df) * (1 - self.args.split)), :]
        valid_df = bags_df.iloc[int(len(bags_df) * (1 - self.args.split)):, :]
        valid_df, test_df = (
            valid_df.iloc[0:len(valid_df) // 2, :],
            valid_df.iloc[len(valid_df) // 2:, :]
        )
        return train_df, valid_df, test_df

    def _load_split_data(self, split_path, split_name):
        print(f'Loading {split_name} data... (mp={self.args.use_mp})...')
        start_time = time.time()
        data = load_data(split_path, self.args)
        print(f'DONE (Took {(time.time() - start_time):.1f}s)')
        return data

    def _log_initial_metrics(self):
        initial_metrics = self.trainer.valid(self.valid_data)
        print(f'\nInitial Metrics')
        print_table(initial_metrics)
        initial_metrics_file_path = os.path.join(self.save_path, f'initial_results.txt')
        with open(initial_metrics_file_path, 'w') as f:
            json.dump(initial_metrics, f, cls=NumpyFloatValuesEncoder)
        wandb.save(initial_metrics_file_path)

    def _load_epoch_model(self, epoch: int):
        model_save_path = os.path.join(self.save_path, f'{epoch}.pth')
        log_save_path = os.path.join(self.save_path, f'thresholds_{epoch}.txt')
        lambda_parameter_save_path = os.path.join(self.save_path, f'lambda_parameter_{epoch}')

        self.trainer.milnet.load_state_dict(torch.load(model_save_path), strict=True)

        with open(log_save_path, 'r') as f:
            epoch_valid_metrics = json.load(f)
        thresholds_optimal = np.asarray(a=eval(epoch_valid_metrics['thresholds_optimal']), dtype=np.float32)
        report = f'Using thresholds_optimal: {thresholds_optimal}'

        feats_thresholds_optimal = epoch_valid_metrics['feats_thresholds_optimal']
        if feats_thresholds_optimal is not None:
            feats_thresholds_optimal = np.asarray(a=eval(feats_thresholds_optimal), dtype=np.float32)
            report += f' feats_thresholds_optimal: {feats_thresholds_optimal}'

        if hasattr(self.trainer, 'lambda_parameter'):
            report += f' lambda_parameter: {self.trainer.lambda_parameter}'
            self.trainer.lambda_parameter = torch.load(lambda_parameter_save_path)
        print(report)
        return thresholds_optimal, feats_thresholds_optimal

    def _save_epoch_model(
            self,
            thresholds_optimal: list,
            epoch: int,
            auc: float,
            feats_thresholds_optimal=None,
            report_prefix: str = None,
    ):
        model_save_path = os.path.join(self.save_path, f'{epoch}.pth')
        log_save_path = os.path.join(self.save_path, f'thresholds_{epoch}.txt')
        lambda_parameter_save_path = os.path.join(self.save_path, f'lambda_parameter_{epoch}')

        model_report = f'model saved at: {model_save_path}'
        torch.save(self.trainer.milnet.state_dict(), model_save_path)

        thresholds_report = f'threshold: {str(thresholds_optimal)}'
        with open(log_save_path, 'w') as f:
            json.dump({
                'auc': auc,
                'thresholds_optimal': str(thresholds_optimal),
                'feats_thresholds_optimal': str(
                    feats_thresholds_optimal
                ) if feats_thresholds_optimal is not None else None
            }, f)

        lambda_parameter_report = ''
        if hasattr(self.trainer, 'lambda_parameter'):
            lambda_parameter_report = f'lambda_parameter: {self.trainer.lambda_parameter}'
            torch.save(self.trainer.lambda_parameter, lambda_parameter_save_path)

        should_log_report = report_prefix is not None
        if should_log_report:
            print(f'\t[{report_prefix}] {model_report} {thresholds_report} {lambda_parameter_report}')

    def run(self):
        if self.args.for_roc_curve:
            data_mapping = {
                'train': self.train_data,
                'valid': self.valid_data,
                'test': self.test_data,
            }
            roc_data_split = data_mapping[self.args.roc_data_split]
            self.save_path = os.path.join(SAVE_PATH, self.args.dataset,
                                          self.args.roc_run_name)  # CHANGES THE SAVE PATH!
            self._load_epoch_model(self.args.roc_run_epoch)
            self.trainer.test(roc_data_split, thresholds_optimal=None, feats_thresholds_optimal=None)
            return
        best_auc_epochs, least_loss_epochs, lowest_error_plus_loss_epochs, best_acc_epochs = self.run_train()
        self.run_test(best_auc_epochs, least_loss_epochs, lowest_error_plus_loss_epochs, best_acc_epochs)
        self.clean_up(best_auc_epochs, least_loss_epochs, lowest_error_plus_loss_epochs, best_acc_epochs)

    def run_train(self):
        best_auc = 0
        best_auc_epochs = []
        least_loss = None
        least_loss_epochs = []
        lowest_error_plus_loss = None
        lowest_error_plus_loss_epochs = []
        best_acc = 0
        best_acc_epochs = []

        best_feat_auc = 0  # For stopping the runs that are not going well.

        self._log_initial_metrics()
        for epoch in range(1, self.args.num_epochs + 1):
            start_train_epoch_time = time.time()
            epoch_train_metrics = self.trainer.train(self.train_data)
            start_valid_epoch_time = time.time()
            epoch_valid_metrics = self.trainer.valid(self.valid_data)
            end_valid_epoch_time = time.time()

            valid_aucs = epoch_valid_metrics['epoch_valid_aucs']
            thresholds_optimal = epoch_valid_metrics['epoch_valid_thresholds_optimal']
            feats_thresholds_optimal = epoch_valid_metrics['epoch_valid_feats_thresholds_optimal']
            epoch_train_time = int(start_valid_epoch_time - start_train_epoch_time)
            epoch_valid_time = int(end_valid_epoch_time - start_valid_epoch_time)

            wandb.log({
                'epoch': epoch,
                'epoch_train_time': epoch_train_time,
                'epoch_valid_time': epoch_valid_time,
                **to_wandb_format(epoch_train_metrics),
                **to_wandb_format(epoch_valid_metrics),
            })
            print(
                '\rEpoch [%d/%d] time %.1fs train loss: %.4f test loss: %.4f,'
                ' thresholds_optimal: %s, feats_thresholds_optimal: %s, accuracy: %.4f, AUC: ' % (
                    epoch,
                    self.args.num_epochs,
                    epoch_train_time + epoch_valid_time,
                    epoch_train_metrics['epoch_train_loss'],
                    epoch_valid_metrics['epoch_valid_loss'],
                    epoch_valid_metrics['epoch_valid_thresholds_optimal'],
                    epoch_valid_metrics['epoch_valid_feats_thresholds_optimal'],
                    epoch_valid_metrics['epoch_valid_accuracy']
                ) +
                '|'.join('class-{0}>>{1:.4f}'.format(*k) for k in enumerate(valid_aucs))
            )

            if self.trainer.scheduler is not None:
                self.trainer.scheduler.step()

            current_auc = valid_aucs[0]
            current_loss = epoch_valid_metrics['epoch_valid_loss']
            current_error_plus_loss = (current_loss + (1 - epoch_valid_metrics['epoch_valid_accuracy']))
            current_acc = epoch_valid_metrics['epoch_valid_accuracy']

            report_prefix = ''
            if current_auc >= best_auc:
                report_prefix += '[best auc]'
                if current_auc > best_auc:
                    best_auc_epochs = []
                best_auc = current_auc
                best_auc_epochs.append(epoch)

            if current_acc >= best_acc:
                report_prefix += '[best acc]'
                if current_acc > best_acc:
                    best_acc_epochs = []
                best_acc = current_acc
                best_acc_epochs.append(epoch)

            if least_loss is None or current_loss <= least_loss:
                report_prefix += '[least loss]'
                if least_loss is not None and current_loss < least_loss:
                    least_loss_epochs = []
                least_loss = current_loss
                least_loss_epochs.append(epoch)

            if lowest_error_plus_loss is None or current_error_plus_loss <= lowest_error_plus_loss:
                report_prefix += '[lowest error and loss]'
                if lowest_error_plus_loss is not None and current_error_plus_loss < lowest_error_plus_loss:
                    lowest_error_plus_loss_epochs = []
                lowest_error_plus_loss = current_error_plus_loss
                lowest_error_plus_loss_epochs.append(epoch)

            self._save_epoch_model(
                thresholds_optimal, epoch, current_auc, feats_thresholds_optimal, report_prefix=report_prefix
            )  # +

            if epoch_valid_metrics['epoch_valid_feat_aucs'] is not None:
                current_feat_auc = epoch_valid_metrics['epoch_valid_feat_aucs'][0]
                best_feat_auc = max(best_feat_auc, current_feat_auc)
                if epoch > 10 and best_feat_auc < 0.05:
                    raise Exception(f'epoch: {epoch} | best_feat_auc: {best_feat_auc}. Stopping...')

        # if self.args.arch in ['abmil-paper', 'abmil-gated-paper']:
        #     print(f'discarding [best auc] and [lowest error] epochs')
        #     best_auc_epochs = []
        #     least_loss_epochs = []

        train_metrics = {
            'best_auc': best_auc,
            'best_auc_epochs': best_auc_epochs,
            'best_acc': best_acc,
            'best_acc_epochs': best_acc_epochs,
            'least_loss': least_loss,
            'least_loss_epochs': least_loss_epochs,
            'lowest_error_and_loss': lowest_error_plus_loss,
            'lowest_error_and_loss_epochs': lowest_error_plus_loss_epochs,
        }
        with open(os.path.join(self.save_path, 'train_metrics.json'), 'w') as f:
            json.dump(train_metrics, f)
        print(f'Train Metrics')
        print(json.dumps(train_metrics) + '\n')

        earliest_best_auc_epoch = min(best_auc_epochs, default=None)
        earliest_least_loss_epoch = min(least_loss_epochs, default=None)
        earliest_lowest_error_and_loss_epochs = min(lowest_error_plus_loss_epochs, default=None)
        earliest_best_acc_epoch = min(best_acc_epochs, default=None)

        return (
            [earliest_best_auc_epoch],
            [earliest_least_loss_epoch],
            [earliest_lowest_error_and_loss_epochs],
            [earliest_best_acc_epoch]
        )
        # return best_auc_epochs, least_loss_epochs, lowest_error_plus_loss_epochs, best_acc_epochs

    def run_test(self, best_auc_epochs, least_loss_epochs, lowest_error_plus_loss_epochs, best_acc_epochs):
        earliest_best_auc_epoch = min(best_auc_epochs, default=None)
        earliest_least_loss_epoch = min(least_loss_epochs, default=None)
        earliest_lowest_error_and_loss_epochs = min(lowest_error_plus_loss_epochs, default=None)
        earliest_best_acc_epoch = min(best_acc_epochs, default=None)

        last_epoch = self.args.num_epochs
        special_epochs = [
            (earliest_best_auc_epoch, 'best_auc'),
            (earliest_least_loss_epoch, 'least_loss'),
            (earliest_lowest_error_and_loss_epochs, 'lowest_error_loss'),
            (last_epoch, 'last_epoch'),
            (earliest_best_acc_epoch, 'best_acc')
        ]
        special_epochs = [x for x in special_epochs if x[0] is not None]
        for epoch, plot_prefix in special_epochs:
            start_test_epoch_time = time.time()
            thresholds_optimal, feats_thresholds_optimal = self._load_epoch_model(epoch)
            epoch_test_metrics = self.trainer.test(self.test_data, thresholds_optimal, feats_thresholds_optimal)
            res = replace_key_names(d=epoch_test_metrics, old_term='epoch', new_term=plot_prefix)
            epoch_test_time = int(time.time() - start_test_epoch_time)
            wandb.log({
                'epoch': epoch,
                'epoch_test_time': epoch_test_time,
                **to_wandb_format(res),
            })
            print('\r', end='')
            print_table({
                'epoch_test_time': epoch_test_time,
                **epoch_test_metrics
            })
            print()

    def clean_up(self, best_auc_epochs, least_loss_epochs, lowest_error_plus_loss_epochs, best_acc_epochs):
        special_epochs = list(
            set(best_auc_epochs + least_loss_epochs + lowest_error_plus_loss_epochs + best_acc_epochs)
        )
        wanted_epochs = []
        for epoch in special_epochs:
            # wanted_epochs.extend(list(range(epoch - 5, epoch + 6)))
            wanted_epochs.extend(list(range(epoch - 0, epoch + 1)))

        for epoch in range(1, self.args.num_epochs + 1):
            if epoch not in wanted_epochs:
                delete_files_for_epoch(self.save_path, epoch)


class DSMILTrainer(Trainer):
    def _get_milnet(self):
        i_classifier = dsmil.FCLayer(in_size=self.args.feats_size, out_size=self.args.num_classes).cuda()
        b_classifier = dsmil.BClassifier(
            input_size=self.args.feats_size,
            output_class=self.args.num_classes,
            dropout_v=self.args.dropout_node,
            nonlinear=self.args.non_linearity
        ).cuda()
        return dsmil.MILNet(i_classifier, b_classifier).cuda()

    def _run_model(self, bag_feats, bag_label) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ins_prediction, bag_prediction, attentions = self.milnet(bag_feats)
        max_prediction, _ = torch.max(ins_prediction, 0)
        bag_loss = self.criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = self.criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        loss = 0.5 * bag_loss + 0.5 * max_loss

        with torch.no_grad():
            if self.args.average:
                # A better place for this would be in the BClassifier!
                bag_prediction = (0.5 * torch.sigmoid(max_prediction) +
                                  0.5 * torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()
            else:
                bag_prediction = (0.0 * torch.sigmoid(max_prediction) +
                                  1.0 * torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()

        return bag_prediction, loss, attentions

    def __str__(self):
        return 'DSMIL'


class DSMILWithInitPTHTrainer(DSMILTrainer):
    def _load_init_weights(self):
        state_dict_weights = torch.load('init.pth')
        try:
            self.milnet.load_state_dict(state_dict_weights, strict=False)
        except Exception as e:
            print(f'Exception during loading init.pth: {e}')
            del state_dict_weights['b_classifier.v.1.weight']
            del state_dict_weights['b_classifier.v.1.bias']
            self.milnet.load_state_dict(state_dict_weights, strict=False)
        print(f'Loaded init.pth weights successfully')


class DSMILWithPaperConfigTrainer(DSMILTrainer):
    def __init__(self, args):
        args = self._override_args(args)
        self.args = args
        super().__init__(args)

    @staticmethod
    def _override_args(args):
        args.lr = 1e-4
        args.optimizer = 'adam'
        args.scheduler = None
        args.average = False  # doesn't really matter, because we replaced it in the _run_model method
        args.weight_init = None
        return args

    def _run_model(self, bag_feats, bag_label) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ins_prediction, bag_prediction, attentions = self.milnet(bag_feats)
        max_prediction, _ = torch.max(ins_prediction, 0)
        bag_loss = self.criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = self.criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        if self.milnet.training:
            loss = 0.5 * bag_loss + 0.5 * max_loss
        else:
            loss = 1.0 * bag_loss + 0.0 * max_loss

        with torch.no_grad():
            if self.milnet.training:
                # A better place for this would be in the BClassifier!
                bag_prediction = (0.5 * torch.sigmoid(max_prediction) +
                                  0.5 * torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()
            else:
                bag_prediction = (0.0 * torch.sigmoid(max_prediction) +
                                  1.0 * torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()

        return bag_prediction, loss, attentions


class DSMILWithGithubConfigTrainer(DSMILTrainer):
    # TODO The actual DSMIL Github Implementation:
    #   validation: 0.5 * torch.sigmoid(max_prediction) + 0.5 * torch.sigmoid(bag_prediction)  (train_tcga.py)
    #   test:       0.0 * torch.sigmoid(max_prediction) + 1.0 * torch.sigmoid(bag_prediction) (testing_c16.py)
    #  but here we do the former for both.
    def __init__(self, args):
        args = self._override_args(args)
        self.args = args
        super().__init__(args)

    @staticmethod
    def _override_args(args):
        args.lr = 2e-4
        args.weight_decay = 5e-3
        args.dropout_patch = 0
        args.dropout_node = 0
        args.non_linearity = 1
        args.average = True
        args.weight_init = None
        return args


class DSMILWithGithubMILConfigTrainer(DSMILTrainer):
    def __init__(self, args):
        args = self._override_args(args)
        self.args = args
        super().__init__(args)

    @staticmethod
    def _override_args(args):
        args.lr = 2e-4
        args.weight_decay = 5e-3
        args.dropout_patch = 0
        args.dropout_node = 0
        args.non_linearity = 1
        args.optimizer = 'adam'
        args.eta_min = 0
        args.average = True
        args.weight_init = None
        args.scheduler = 'cosine'
        return args


class ABMILTrainer(Trainer):
    def _get_milnet(self):
        return abmil.Attention(feats_size=self.args.feats_size).cuda()

    def _run_model(self, bag_feats, bag_label) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bag_feats = bag_feats.view(-1, self.args.feats_size)
        bag_prediction, _, attentions = self.milnet(bag_feats)
        bag_label = bag_label.view(1, -1)
        bag_loss = self.criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        loss = bag_loss

        bag_prediction = bag_prediction.squeeze().detach().cpu().numpy()
        return bag_prediction, loss, attentions

    def __str__(self):
        return 'ABMIL'


class ABMILGatedTrainer(ABMILTrainer):
    def _get_milnet(self) -> nn.Module:
        return abmil.GatedAttention(feats_size=self.args.feats_size).cuda()

    def __str__(self):
        return 'ABMILGated'


class ABMILWithPaperConfigsTrainer(ABMILTrainer):
    """
    ABMIL Official Params According to the Supplementary Materials.
    https://proceedings.mlr.press/v80/ilse18a/ilse18a-supp.pdf
    """

    def __init__(self, args):
        args = self._override_args(args)
        self.args = args
        super().__init__(args)

    @staticmethod
    def _override_args(args):
        args.weight_init = 'xavier_uniform'
        args.scheduler = None
        args.num_epochs = 40

        if args.dataset in MIL_DATASETS:
            mi_net_hyperparams = {
                'musk1': (0.0005, 0.005),
                'musk2': (0.0005, 0.03),
                'fox': (0.0005, 0.005),
                'tiger': (0.0001, 0.01),
                'elephant': (0.0001, 0.005),
            }
            args.optimizer = 'sgd'
            args.lr, args.weight_decay = mi_net_hyperparams[args.dataset]

        elif args.dataset in HISTOPATHOLOGY_DATASETS:
            args.optimizer = 'adam'
            args.lr, args.weight_decay = 0.0001, 0.0005
        return args

    def _get_optimizer(self) -> optim.Optimizer:
        if self.args.dataset in MIL_DATASETS:
            return torch.optim.SGD(
                params=self.milnet.parameters(),
                lr=self.args.lr,  # set in _override_args
                momentum=0.9,
                weight_decay=self.args.weight_decay  # set in _override_args
            )
        elif self.args.dataset in HISTOPATHOLOGY_DATASETS:
            return torch.optim.Adam(
                params=self.milnet.parameters(),
                lr=self.args.lr,  # set in _override_args
                betas=(0.9, 0.999),
                weight_decay=self.args.weight_decay  # set in _override_args
            )
        else:
            raise NotImplementedError(
                f'ABMILWithPaperConfigsTrainer only accepts histopathology (Camelyon16, TCGA)'
                f' and classical MIL datasets: {HISTOPATHOLOGY_DATASETS + MIL_DATASETS}'
            )

    def _run_model(self, bag_feats, bag_label) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bag_feats = bag_feats.view(-1, self.args.feats_size)
        # The following is what the official ABMIL code does - (Inefficient, calls forward twice)
        loss, attentions = self.milnet.calculate_objective(bag_feats, bag_label)
        _, bag_prediction = self.milnet.calculate_classification_error(bag_feats, bag_label)
        bag_prediction = bag_prediction.squeeze().detach().cpu().numpy()
        return bag_prediction, loss, attentions

    def __str__(self):
        return 'ABMILWithPaperConfigs'


class ABMILGatedWithPaperConfigsTrainer(ABMILWithPaperConfigsTrainer):
    def _get_milnet(self) -> nn.Module:
        return abmil.GatedAttention(feats_size=self.args.feats_size).cuda()

    def __str__(self):
        return 'ABMILGatedWithPaperConfigs'


class LambdaTrainer(Trainer):
    def __init__(self, args):
        self.args = args
        self.lambda_parameter = self._get_lambda_parameter()
        super().__init__(args)

    def _get_lambda_parameter(self):
        lambda_parameter = torch.tensor(0.5, requires_grad=self.args.soft_average, device='cuda')
        print('lambda_parameter.requires_grad:', lambda_parameter.requires_grad)
        lambda_parameter.data.clamp_(0, 1)
        return lambda_parameter

    def _get_optimizer(self) -> optim.Optimizer:
        try:
            optimizer_cls = OPTIMIZERS[self.args.optimizer]
        except KeyError:
            raise Exception(f'Optimizer not found. Given: {self.args.optimizer}, Have: {OPTIMIZERS.keys()}')

        print(
            f'Optimizer {self.args.optimizer} with lr={self.args.lr}, betas={(0.5, 0.9)}, wd={self.args.weight_decay}'
        )
        return optimizer_cls(
            # params=[self.lambda_parameter] + list(self.milnet.parameters()),
            params=[
                {'params': self.lambda_parameter, 'lr': self.args.lr * self.args.lambda_lr_multiplier},
                {'params': self.milnet.parameters()}
            ],
            lr=self.args.lr,
            betas=(0.5, 0.9),
            weight_decay=self.args.weight_decay
        )

    def _run_model(self, bag_feats, bag_label) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        ins_prediction, bag_prediction, attentions = self.milnet(bag_feats)
        if len(ins_prediction.shape) == 2:
            max_prediction, _ = torch.max(ins_prediction, 0)
        else:
            max_prediction, _ = torch.max(ins_prediction, 1)  # milformer
        bag_loss = self.criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = self.criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        loss = self.lambda_parameter * bag_loss + (1 - self.lambda_parameter) * max_loss

        # print('devices:', self.lambda_parameter.get_device(), max_prediction.get_device())

        with torch.no_grad():
            bag_prediction = (
                    (1 - self.lambda_parameter) * torch.sigmoid(max_prediction) +
                    self.lambda_parameter * torch.sigmoid(bag_prediction)
            ).squeeze().cpu().numpy()

        # return bag_prediction, loss, attentions
        return bag_prediction, loss, ins_prediction

    def train(self, data):
        res = super().train(data)
        res = {
            **res,
            'small_lambda': self.lambda_parameter
        }
        return res

    def _after_run_model_in_training_mode(self):
        self.optimizer.step()
        self.lambda_parameter.data.clamp_(0, 1)

    def __str__(self):
        return f'Lambda_sa{self.args.soft_average}'


class LambdaDSMILTrainer(LambdaTrainer):
    def _get_milnet(self) -> nn.Module:
        i_classifier = dsmil.FCLayer(in_size=self.args.feats_size, out_size=self.args.num_classes).cuda()
        b_classifier = dsmil.BClassifier(
            input_size=self.args.feats_size,
            output_class=self.args.num_classes,
            dropout_v=self.args.dropout_node,
            nonlinear=self.args.non_linearity
        ).cuda()
        return dsmil.MILNet(i_classifier, b_classifier).cuda()

    def __str__(self):
        return 'LambdaDSMIL'


class MILFormer(LambdaTrainer):
    def _get_milnet(self) -> nn.Module:
        i_classifier = milformer.FCLayer(in_size=self.args.feats_size,
                                                out_size=self.args.num_classes).cuda()

        c = copy.deepcopy
        attn = milformer.MultiHeadedAttention(self.args.num_heads, self.args.feats_size).cuda()
        ff = milformer.PositionwiseFeedForward(
            self.args.feats_size, self.args.feats_size * self.args.mlp_multiplier, self.args.encoder_dropout
        ).cuda()
        b_classifier = milformer.BClassifier(
            milformer.Encoder(
                milformer.EncoderLayer(self.args.feats_size, c(attn), c(ff), self.args.encoder_dropout,
                                              self.args.k, self.args.random_patch_share), 1),
            self.args.num_classes, self.args.feats_size
        ).cuda()
        milnet = milformer.MILNet(i_classifier, b_classifier).cuda()

        if self.args.weight_init == 'xavier_normal':
            for p in milnet.parameters():
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)
        elif self.args.weight_init == 'xavier_uniform':
            for p in milnet.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        elif self.args.weight_init == 'trunc_normal':
            for p in milnet.parameters():
                if p.dim() > 1:
                    nn.init.trunc_normal_(p)

        return milnet

    def _run_model(self, bag_feats, bag_label) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # bag_prediction, loss, A = super()._run_model(bag_feats, bag_label)
        # attentions = A.mean(dim=1).mean(dim=2).reshape(-1, 1)
        # return bag_prediction, loss, attentions

        bag_prediction, loss, ins_prediction = super()._run_model(bag_feats, bag_label)
        ins_prediction = ins_prediction.view(-1, 1)
        # return bag_prediction, loss, ins_prediction
        return bag_prediction, loss, torch.sigmoid(ins_prediction)

    def __str__(self):
        return f'MILFormer_k{self.args.k}_sa{self.args.soft_average}'


class MILFormerForSweepTrainer(MILFormer):
    """
    Wandb Sweep doesn't allow conditional parameters. We have to override the conditional params manually.
    """

    def __init__(self, args):
        args = self._override_args(args)
        self.args = args
        super().__init__(args)

    @staticmethod
    def _override_args(args):
        if args.scheduler == 'cosinewarmup':
            print(f'For milformer cosinewarmup, multiplying lr by 1.5')
            args.lr = 1.5 * args.lr

        lr_degree = abs(int(f"{args.lr:e}".split("e")[-1]))
        if lr_degree == 4:
            args.lambda_lr_multiplier = 1.0
        elif lr_degree == 3:
            args.lambda_lr_multiplier = 0.1
        elif lr_degree == 2:
            args.lambda_lr_multiplier = 0.01
        print(f'lr_degree: {lr_degree} > Overriding lambda_lr_multiplier to {args.lambda_lr_multiplier}')

        return args


class MILFormerWithSweepResultsTrainer(MILFormer):
    def __init__(self, args):
        args = self._override_args(args)
        self.args = args
        super().__init__(args)

    @staticmethod
    def _override_args(args):
        args.lr = 2e-2
        args.num_epochs = 200
        args.weight_init = 'trunc_normal'
        args.optimizer = 'adamw'
        args.scheduler = 'cosine'
        args.encoder_dropout = 0.1
        args.lambda_lr_multiplier = 0.1
        args.weight_decay = 5e-3
        # args.soft_average = True
        args.num_heads = 4
        # args.k = 200
        # args.random_patch_share = 0.5

        lr_degree = abs(int(f"{args.lr:e}".split("e")[-1]))
        if lr_degree == 4:
            args.lambda_lr_multiplier = 1.0
        elif lr_degree == 3:
            args.lambda_lr_multiplier = 0.1
        elif lr_degree == 2:
            args.lambda_lr_multiplier = 0.01
        print(f'lr_degree: {lr_degree} > Overriding lambda_lr_multiplier to {args.lambda_lr_multiplier}')

        return args



class MeanPoolingTrainer(DSMILTrainer):
    def _run_model(self, bag_feats, bag_label) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ins_prediction, bag_prediction, attentions = self.milnet(bag_feats)
        mean_prediction = torch.mean(ins_prediction, 0)
        mean_loss = self.criterion(mean_prediction.view(1, -1), bag_label.view(1, -1))
        loss = mean_loss

        with torch.no_grad():
            bag_prediction = torch.sigmoid(mean_prediction).squeeze().cpu().numpy()

        return bag_prediction, loss, ins_prediction

    def __str__(self):
        return f'MeanPooling_sa{self.args.soft_average}'


class MeanPoolingWithDSMILPaperConfigsTrainer(MeanPoolingTrainer):
    """
    MeanPoolingTrainer with DSMIL Paper Hyperparams
    """

    def __init__(self, args):
        args = self._override_args(args)
        self.args = args
        super().__init__(args)

    @staticmethod
    def _override_args(args):
        args.lr = 1e-4
        args.optimizer = 'adam'
        args.scheduler = None
        args.average = False  # doesn't really matter, because we replaced it in the _run_model method
        args.weight_init = None
        return args

    def __str__(self):
        return f'MeanPooling_paper_sa{self.args.soft_average}'


class MaxPoolingTrainer(LambdaDSMILTrainer):
    def _get_lambda_parameter(self):
        lambda_parameter = torch.tensor(0.0, requires_grad=False, device='cuda')
        print('lambda_parameter.requires_grad:', lambda_parameter.requires_grad)
        lambda_parameter.data.clamp_(0, 1)
        return lambda_parameter

    def __str__(self):
        return 'MaxPooling'


class MaxPoolingWithDSMILPaperConfigsTrainer(LambdaDSMILTrainer):
    def __init__(self, args):
        args = self._override_args(args)
        self.args = args
        super().__init__(args)

    @staticmethod
    def _override_args(args):
        args.lr = 1e-4
        args.optimizer = 'adam'
        args.scheduler = None
        args.average = False  # doesn't really matter, because we replaced it in the _run_model method
        args.weight_init = None
        return args

    def _get_lambda_parameter(self):
        lambda_parameter = torch.tensor(0.0, requires_grad=False, device='cuda')
        print('lambda_parameter.requires_grad:', lambda_parameter.requires_grad)
        lambda_parameter.data.clamp_(0, 1)
        return lambda_parameter

    def __str__(self):
        return 'MaxPooling_paper'


def get_run_name(args, trainer):
    base_run_name = f'{args.embedding}_{str(trainer)}'
    save_path_base = os.path.join(SAVE_PATH, args.dataset)

    previous_run_dirs = [d for d in os.listdir(save_path_base) if os.path.isdir(os.path.join(save_path_base, d))]
    filtered_dirs = [dir for dir in previous_run_dirs if re.match(rf'^{base_run_name}_\d+$', dir)]
    run_numbers = [int(re.findall(r'\d+$', dir)[0]) for dir in filtered_dirs]

    last_run_number = max(run_numbers) if run_numbers else 0
    run_number = last_run_number + 1

    run_name = f'{base_run_name}_{run_number}'
    return run_name


def validate_args(args):
    args.use_mp = bool(args.use_mp)
    args.soft_average = bool(args.soft_average)
    args.for_roc_curve = (
            args.roc_run_name is not None and args.roc_run_epoch is not None
    )

    mil_dataset_to_num_feats_mapping = {
        'musk1': 166,
        'musk2': 166,
        'elephant': 230,
        'fox': 230,
        'tiger': 230
    }
    if args.dataset in mil_dataset_to_num_feats_mapping.keys():
        args.feats_size = mil_dataset_to_num_feats_mapping[args.dataset]
        print(f'Setting feats_size to {args.feats_size} for {args.dataset}')

    return args


def main():
    parser = argparse.ArgumentParser('MILFormer Trainer', parents=[get_args_parser()], add_help=False)
    args = parser.parse_args()
    args = validate_args(args)

    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    arch_registry = {
        'dsmil': DSMILTrainer,
        'dsmil-paper': DSMILWithPaperConfigTrainer,
        'dsmil-github': DSMILWithGithubConfigTrainer,
        'dsmil-github-mil': DSMILWithGithubMILConfigTrainer,
        'abmil': ABMILTrainer,
        'abmil-gated': ABMILGatedTrainer,
        'abmil-paper': ABMILWithPaperConfigsTrainer,
        'abmil-gated-paper': ABMILGatedWithPaperConfigsTrainer,
        'mean-pooling': MeanPoolingTrainer,
        'mean-pooling-paper': MeanPoolingWithDSMILPaperConfigsTrainer,
        'max-pooling': MaxPoolingTrainer,
        'max-pooling-paper': MaxPoolingWithDSMILPaperConfigsTrainer,
        'milformer': MILFormer,
        'milformer-sweep': MILFormerForSweepTrainer,
        'milformer-sweep-results': MILFormerWithSweepResultsTrainer,
    }
    try:
        trainer = arch_registry[args.arch](args)
    except KeyError as e:
        raise Exception(f'Invalid Architecture: {args.arch} | Choose from: {arch_registry.keys()}')

    if args.wandb_run is None:
        # args.wandb_run = get_run_name(args, trainer)
        # print(f'wandb_run name arg is not specified. defaulted to {args.wandb_run}')
        print(f'No wandb name generated by us.')

    wandb.init(
        project=f"MIL_{args.dataset}",
        config={**vars(args)},
        mode='disabled' if args.for_roc_curve else None
        # settings=wandb.Settings(disable_git=True, save_code=False),
    )
    print(f'*** Run Config *** ')
    pretty_print({**vars(args)})

    runner = Runner(args, trainer)
    runner.run()
    wandb.finish()


if __name__ == '__main__':
    main()
