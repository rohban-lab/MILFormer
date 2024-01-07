### most the credit to https://github.com/binli123/dsmil-wsi (we did some changes for our convenience and working with some other backbones)
import time
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from pprint import pprint
import argparse, copy, glob, os, sys, warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ViT_B_16_Weights
)
import torchvision.transforms.functional as VF
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.utils import shuffle

import dsmil as mil
from utils import check_layers

DATASETS_PATH = '../datasets'
EMBEDDINGS_PATH = 'embeddings'


class BagDataset:
    def __init__(self, files_list: List[str], transform=None, patch_labels_dict: Dict[str, int] = None):
        if patch_labels_dict is None:
            patch_labels_dict = {}

        self.files_list = files_list
        self.transform = transform
        self.patch_labels = patch_labels_dict

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img)

        patch_address = os.path.join(
            *temp_path.split(os.path.sep)[-3:]  # class_name/bag_name/patch_name.jpeg
        )
        label = self.patch_labels.get(patch_address, -1)  # TCGA doesn't have patch labels, set -1 to ignore

        patch_name = Path(temp_path).stem
        # Camelyon16 Patch Name Convention: {row}_{col}-17.jpeg > 116_228-17.jpeg
        # TCGA       Patch Name Convention: {row}_{col}.jpeg    > 116_228-17.jpeg
        row, col = patch_name.split('-')[0].split('_')
        position = np.asarray([int(row), int(col)])

        sample = {
            'input': img,
            'label': label,
            'position': position
        }

        if self.transform:
            sample = self.transform(sample)
        return sample


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['input']
        img = VF.resize(img, self.size)
        return {
            **sample,
            'input': img
        }


class ToTensor:
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)

        label = sample['label']
        assert isinstance(label, int), f"A sample label should be of type int, but {type(label)} received."
        return {
            **sample,
            'label': torch.tensor(label),
            'input': img
        }


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def bag_dataset(args, patches: List[str], patch_labels_dict: dict = None) -> Tuple[DataLoader, int]:
    """
    Create a bag dataset and its corresponding data loader.

    This function creates a bag dataset from the provided list of patch file paths and prepares a data loader to access
    the data in batches. The bag dataset is expected to contain bag-level data, where each bag is represented as a
    collection of instances.

    Args:
        args (object): An object containing arguments or configurations for the data loader setup.
        patches (List[str]): A list of file paths representing patches.
        patch_labels_dict (dict): A dict in the form {patch_name: patch_label}

    Returns:
        tuple: A tuple containing two elements:
            - dataloader (torch.utils.data.DataLoader): The data loader to access the bag dataset in batches.
            - dataset_size (int): The total number of bags (patches) in the dataset.
    """
    transforms = [ToTensor()]
    # TODO remove this
    if args.backbone == 'vit-b-16':
        # print(f'Resizing images from 256*256 to 224*224 to use ViT-B-16 ImageNet features.')
        transforms.insert(0, Resize((224, 224)))
    transformed_dataset = BagDataset(
        files_list=patches,
        transform=Compose(transforms),
        patch_labels_dict=patch_labels_dict
    )
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)


def compute_feats(
        args, bags_list: List[str], embedder: nn.Module, save_path: str,
        patch_labels_dict: dict = None, magnification: str = 'single'
):
    """
    Compute features for bag data using the provided embedder.

    This function takes bag data in the form of a list of bags and computes features for every patch of each bag using
     the specified embedder. For each bag, a file named
      bag_name.csv [feature_1, ..., feature_511, position, label] will be saved.
      Each row is for a patch of that bag.

    Args:
        args (object): An object containing additional arguments or configuration for the feature computation.
        bags_list (list): A list of bags, where each bag is represented as a path to a directory,
                            where any jpg/jpeg file in this path is treated as a patch for this bag.
        embedder (Callable): A function that takes a patch (instance of a bag) as input and returns its corresponding
        feature vector.
       save_path (str): The path to save the computed features.
        patch_labels_dict (dict): A dictionary in the format {patch_address: patch_label}, for c16.
        magnification (str, optional): The magnification level for the feature computation. Default is 'single'.

    Returns:
        None
    """
    embedder.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor

    for i in tqdm(range(num_bags)):
        feats_list = []
        feats_labels = []
        feats_positions = []

        patches = None
        if magnification in ['single', 'low']:
            patches = glob.glob(os.path.join(bags_list[i], '*.jpg')) + \
                      glob.glob(os.path.join(bags_list[i], '*.jpeg'))
        elif magnification == 'high':
            patches = glob.glob(os.path.join(bags_list[i], '*' + os.sep + '*.jpg')) + \
                      glob.glob(os.path.join(bags_list[i], '*' + os.sep + '*.jpeg'))

        dataloader, bag_size = bag_dataset(args, patches, patch_labels_dict)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda()
                feats, classes = embedder(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
                batch_labels = batch['label']
                feats_labels.extend(np.atleast_1d(batch_labels.squeeze().tolist()).tolist())
                feats_positions.extend(batch['position'])

                tqdm.write(
                    '\r Computed: {}/{} -- {}/{}'.format(i + 1, num_bags, iteration + 1, len(dataloader)), end=''
                )

        if len(feats_list) == 0:
            print('No valid patch extracted from: ' + bags_list[i])
        else:
            df = pd.DataFrame(feats_list, dtype=np.float32)
            df['label'] = feats_labels if patch_labels_dict is not None else np.nan
            df['position'] = feats_positions if patch_labels_dict is not None else None

            split_name, class_name, bag_name = bags_list[i].split(os.path.sep)[-3:]
            csv_directory = os.path.join(save_path, split_name, class_name)
            csv_file = os.path.join(csv_directory, bag_name)
            os.makedirs(csv_directory, exist_ok=True)
            df_save_path = os.path.join(csv_file + '.csv')
            df.to_csv(df_save_path, index=False, float_format='%.4f')


def compute_tree_feats(args, bags_list: List[str], embedder_low: nn.Module, embedder_high: nn.Module, save_path: str):
    # TODO This method has not been tested! (we're not there yet)
    """
    Refer to the `compute_feats` method. The only difference is that this computes features for tree structures.
    """
    embedder_low.eval()
    embedder_high.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    with torch.no_grad():
        for i in range(num_bags):
            low_patches = glob.glob(os.path.join(bags_list[i], '*.jpg')) + \
                          glob.glob(os.path.join(bags_list[i], '*.jpeg'))
            feats_list = []
            feats_tree_list = []
            dataloader, bag_size = bag_dataset(args, low_patches)
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda()
                feats, classes = embedder_low(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
            for idx, low_patch in enumerate(low_patches):
                high_folder = os.path.dirname(low_patch) + os.sep + os.path.splitext(os.path.basename(low_patch))[0]
                high_patches = glob.glob(high_folder + os.sep + '*.jpg') + \
                               glob.glob(high_folder + os.sep + '*.jpeg')
                if len(high_patches) == 0:
                    pass
                else:
                    for high_patch in high_patches:
                        img = Image.open(high_patch)
                        img = VF.to_tensor(img).float().cuda()
                        feats, classes = embedder_high(img[None, :])

                        if args.tree_fusion == 'fusion':
                            feats = feats.cpu().numpy() + 0.25 * feats_list[idx]
                        elif args.tree_fusion == 'cat':
                            feats = np.concatenate((feats.cpu().numpy(), feats_list[idx][None, :]), axis=-1)

                        feats_tree_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i + 1, num_bags, idx + 1, len(low_patches)))
            if len(feats_tree_list) == 0:
                print('No valid patch extracted from: ' + bags_list[i])
            else:
                df = pd.DataFrame(feats_tree_list)
                os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
                df_save_path = os.path.join(
                    save_path,
                    bags_list[i].split(os.path.sep)[-2],  # class name
                    bags_list[i].split(os.path.sep)[-1] + '.csv'  # bag name
                )
                df.to_csv(df_save_path, index=False, float_format='%.4f')
            print('\n')


def get_args_parser():
    parser = argparse.ArgumentParser(description='WSI Patch Embedder')
    parser.add_argument('--embedder', default='SimCLR', type=str, help='Embedder to ba used for feature computation')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size of dataloader [128]')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of threads for dataloader')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--backbone', default='resnet18', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'vit-b-16'],
                        help='Embedder backbone [resnet18]')
    parser.add_argument('--norm_layer', default='instance', type=str, choices=['instance', 'batch'],
                        help='Normalization layer [instance]')
    parser.add_argument('--magnification', default='single', type=str, choices=['single', 'high', 'low', 'tree'],
                        help='Magnification to compute features. Use `tree` for multiple magnifications. '
                             'Use `high` if patches are cropped for multiple resolution and only process higher level,'
                             ' `low` for only processing lower level.')
    parser.add_argument('--weights', default=None, type=str, help='Path to the pretrained embedder weights')
    parser.add_argument('--weights_high', default=None, type=str,
                        help='Path to the pretrained embedder weights for high magnification')
    parser.add_argument('--weights_low', default=None, type=str,
                        help='Path to the pretrained embedder weights for low magnification')
    parser.add_argument('--tree_fusion', default='cat', type=str, choices=['cat', 'fusion'],
                        help='Fusion method for high and low mag features in a tree method [cat|fusion]')
    parser.add_argument('--dataset', default='camelyon16', type=str,
                        help='Dataset folder name [DATASET_PATH/args.dataset]')
    parser.add_argument('--num_processes', default=1, type=int,
                        help='[Not yet implemented] Number of processes for parallel feature computation.')
    return parser


def validate_args(args):
    if 'ImageNet' in [args.weights_high, args.weights_low, args.weights] and args.norm_layer != 'batch':
        raise ValueError('Please use batch normalization for ImageNet feature')

    if args.magnification == 'tree':
        if args.weights_high is None:
            raise ValueError(
                'Specify the path to pretrained weights of high magnification with --weights-high argument.'
            )
        if args.weights_low is None:
            raise ValueError(
                'Specify the path to pretrained weights of low magnification with --weights-low argument.'
            )

    if (args.norm_layer == 'instance' and
            'simclr' not in args.embedder.lower() and
            'imagenet' not in args.embedder.lower()
    ):
        warnings.warn(
            'norm_layer is set to InstanceNorm2D (by default) (As it is used by DSMIL SimCLR implementation).\n'
            'Are you sure that your pretrained model is also using InstanceNorm2D? '
        )

    if ('simclr' not in args.embedder.lower() and
            args.norm_layer != 'batch'
    ):
        warnings.warn(
            'DSMIL official embedder weights require Instance2D Norm Layer'
        )


def get_embedder_backbone(args):
    pretrain = None
    norm = None
    if args.norm_layer == 'instance':
        norm = nn.InstanceNorm2d
        pretrain = False
    elif args.norm_layer == 'batch':
        norm = nn.BatchNorm2d
        pretrain = args.weights == 'ImageNet'

    # Resnet uses AdaptiveAvgPool2d to handle different image sizes
    # TODO what norm_layer should we use for ViT?
    registry = {
        'resnet18': (models.resnet18, 512, ResNet18_Weights.DEFAULT, {'norm_layer': norm}),
        'resnet34': (models.resnet34, 512, ResNet34_Weights.DEFAULT, {'norm_layer': norm}),
        'resnet50': (models.resnet50, 2048, ResNet50_Weights.DEFAULT, {'norm_layer': norm}),
        'resnet101': (models.resnet101, 2048, ResNet101_Weights.DEFAULT, {'norm_layer': norm}),
        'vit-b-16': (models.vit_b_16, 768, ViT_B_16_Weights.DEFAULT, {'image_size': 224})
    }
    model_factory, num_feats, weights, kwargs = registry.get(args.backbone)
    model = model_factory(
        weights=weights if pretrain else None,
        **kwargs
    )

    for param in model.parameters():
        param.requires_grad = False

    if isinstance(model, models.ResNet):
        model.fc = nn.Identity()
    if isinstance(model, models.VisionTransformer):
        model.heads = nn.Identity()

    return model, num_feats


def get_embedder(args, backbone: nn.Module, num_feats: int) -> Tuple[nn.Module, Optional[nn.Module]]:
    if args.magnification == 'tree':
        embedder_high = mil.IClassifier(backbone, num_feats, output_class=args.num_classes).cuda()
        embedder_low = mil.IClassifier(copy.deepcopy(backbone), num_feats, output_class=args.num_classes).cuda()

        if 'ImageNet' in [args.weights_high, args.weights_low, args.weights]:
            print('Using ImageNet features.')
        else:
            _load_model_weights(args, embedder_high, mag='high')
            _load_model_weights(args, embedder_low, mag='low')
            print('Using pretrained features.')

        return embedder_high, embedder_low

    elif args.magnification in ['single', 'high', 'low']:
        embedder = mil.IClassifier(backbone, num_feats, output_class=args.num_classes).cuda()

        if args.weights == 'ImageNet':
            print('Using ImageNet features.')
        else:
            print('Using pretrained features.')
            _load_model_weights(args, embedder)

        return embedder, None


def _load_model_weights(args, embedder: mil.IClassifier, mag=''):
    weights = args.weights
    if mag == 'high':
        weights = args.weights_high
    elif mag == 'low':
        weights = args.weights_low

    if 'SimCLR' in args.embedder:
        state_dict_weights = _get_dsmil_simclr_weights(args, weights)
    elif 'MoCo' in args.embedder:
        state_dict_weights = _get_hossein_moco_weights(args, weights)
    else:
        print('Didnt load any weights for the embedder!')
        return

    check_layers(
        model_state_dict=embedder.state_dict(),
        weights_state_dict=state_dict_weights,
        header='Emebedder',
        align=False
    )

    state_dict_init = embedder.state_dict()
    new_state_dict = OrderedDict()
    print(f'Assigning new layer names...')
    for (loaded_key_i, loaded_val_i), (init_key_i, init_val_i) in zip(state_dict_weights.items(),
                                                                      state_dict_init.items()):
        print(f'Weight key {loaded_key_i} > {init_key_i}')
        new_state_dict[init_key_i] = loaded_val_i

    embedder.load_state_dict(new_state_dict, strict=False)
    print(f'Loaded the embedder weights')

    os.makedirs(os.path.join('embedders', args.embedder, args.dataset), exist_ok=True)
    embedder_name = f'embedder-{mag}.pth' if mag else 'embedder.pth'
    embedder_path = os.path.join('embedders', args.embedder, args.dataset, embedder_name)
    torch.save(new_state_dict, embedder_path)
    print(f'Saved the embedder being used at {embedder_path}')


def _get_hossein_moco_weights(args, weights_path: str):
    weights = torch.load(weights_path)
    weights_state_dict = weights['model_state_dict']
    state_dict_weights = {k: v for k, v in weights_state_dict.items() if "momentum" not in k}
    for i in range(4):
        popped_k, popped_v = state_dict_weights.popitem()
        pprint(f'Popped layer {popped_k} from weights')
    return state_dict_weights


def _get_dsmil_simclr_weights(args, weights: str):
    """
    The simclr/run.py saves its weights at simclr/runs/*/checkpoints/*.pth
    This function loads those weights (taken from the official DSMIL code)
    """
    state_dict_weights = torch.load(weights)
    embedder: mil.IClassifier
    for i in range(4):
        popped_k, popped_v = state_dict_weights.popitem()
        pprint(f'Popped layer {popped_k} from weights')
    return state_dict_weights


def get_bags_path(args):
    if args.magnification in ['tree', 'low', 'high']:
        bags_path = os.path.join(
            DATASETS_PATH, args.dataset, 'pyramid',
            '*',  # train/test/val
            '*',  # classes: 0_normal 1_tumor
            '*'  # bag name
        )
    else:
        bags_path = os.path.join(
            DATASETS_PATH, args.dataset, 'single',
            'fold1',
            '*',  # train/test/val
            '*',  # classes: 0_normal 1_tumor
            '*',  # bag name
        )
    return bags_path


def get_patch_labels_dict(args) -> Optional[Dict[str, int]]:
    patch_labels_path = os.path.join(DATASETS_PATH, args.dataset, 'tile_label_new.csv')

    try:
        labels_df = pd.read_csv(patch_labels_path)
        print(f'Using patch_labels csv file at {patch_labels_path}')
        duplicates = labels_df['slide_name'].duplicated()
        assert not any(duplicates), "There are duplicate patch_names in the {patch_labels_csv} file."
        return labels_df.set_index('slide_name')['label'].to_dict()

    except FileNotFoundError:
        print(f'No patch_labels csv file at {patch_labels_path}')
        return None


def save_class_features(args, save_path):
    """
    Saves a csv [bag_feats_path, bag_label] for each class in each split
     at EMBEDDING_PATH/args.dataset/args.embedder/split/class_name.csv
    Saves a csv [bag_feats_path, bag_label] for the whole dataset
     at EMBEDDING_PATH/args.dataset/args.embedder/args.dataset.csv
    """
    path_to_split_classes = glob.glob(os.path.join(
        save_path,
        '*',  # train/test/val split
        '*' + os.path.sep
    ))

    classes = [item.split(os.path.sep)[-2] for item in path_to_split_classes]
    classes = sorted(list(set(classes)))
    print(f'Classes: {classes}')

    class_df_ls = []
    for path_to_split_class in path_to_split_classes:  # len(path_to_split_classes) = num_splits * num_classes
        bag_csvs = glob.glob(os.path.join(path_to_split_class, '*.csv'))
        class_df = pd.DataFrame(bag_csvs)
        split_name, class_name = path_to_split_class.split(os.path.sep)[-3:-1]

        class_number = classes.index(class_name)
        class_df['label'] = class_number
        class_df_path = os.path.join(EMBEDDINGS_PATH, args.dataset, args.embedder, split_name, class_name + '.csv')
        class_df.to_csv(class_df_path, index=False)
        class_df_ls.append(class_df)
        print(f'Saved class {class_name, class_number} csv [bag_path, bags_label] at {class_df_path}')

    all_df = pd.concat(class_df_ls, axis=0, ignore_index=True)
    all_df = shuffle(all_df)
    all_df_path = os.path.join(save_path, args.dataset + '.csv')
    all_df.to_csv(all_df_path, index=False)
    print(f'Saved dataset csv [bag_path, bags_label] at {all_df_path}')


def main():
    """
    Input:
        - Weights of the embedder model (args.weights, args.weights_low, args.weights_high)
        - Dataset at
    Output:
        - Saves the [cleaned] embedder weights to be reused at test file, when we compute features for the test set
            at EMBEDDINGS_PATH/args.embedder/args.dataset/embedder-{args.mag}.pth
        - Saves a csv [feature_0, ..., feature_511, position, label] for each bag
            at EMBEDDINGS_PATH/args.dataset/args.embedder/
        - Saves a csv [bag_path, bag_label] for each class
            at EMBEDDING_PATH/args.dataset/args.embedder/split/class_name.csv
        - Saves a csv [bag_path, bag_label] for the whole dataset
            at EMBEDDING_PATH/args.dataset/args.embedder/args.dataset.csv
    """
    parser = argparse.ArgumentParser(parents=[get_args_parser()], add_help=False)
    args = parser.parse_args()
    validate_args(args)

    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    backbone, num_feats = get_embedder_backbone(args)

    bags_path = get_bags_path(args)
    print(f'Using bags at {bags_path}')
    feats_path = os.path.join(EMBEDDINGS_PATH, args.dataset, args.embedder)

    os.makedirs(feats_path, exist_ok=True)
    bags_list = glob.glob(bags_path)
    print(f'Number of bags: {len(bags_list)} | Sample Bag: {bags_list[0]}')

    patch_labels_dict = get_patch_labels_dict(args)

    start_time = time.time()
    if args.magnification == 'tree':
        embedder_high, embedder_low = get_embedder(args, backbone, num_feats)
        compute_tree_feats(args, bags_list, embedder_low, embedder_high, feats_path)
    else:
        embedder, _ = get_embedder(args, backbone, num_feats)
        compute_feats(args, bags_list, embedder, feats_path, patch_labels_dict, args.magnification)

    print(f'Took {time.time() - start_time} seconds to compute feats')
    save_class_features(args, feats_path)


if __name__ == '__main__':
    main()
