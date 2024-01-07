import time, sys, argparse, os, glob

sys.path.append('/opt/ASAP/bin')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
from openslide import OpenSlide
import multiresolutionimageinterface as mir
from PIL import Image, ImageFilter
from collections import OrderedDict
from skimage import exposure, io, img_as_ubyte, transform
import copy
import matplotlib.cm as cm
# import plotly.express as px
from tqdm import tqdm

import dsmil
import milformer

from utils import check_layers, load_data


class BagDataset:
    def __init__(self, csv_file, transform=None):
        self.patch_paths = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        path = self.patch_paths[idx]
        img = Image.open(path)

        # patch_name = os.path.basename(path)
        # row, col = patch_name.split('-')[0].split('_')
        patch_name = os.path.splitext(os.path.basename(path))[0]
        row, col = patch_name.split('_')
        img_pos = np.asarray([int(row), int(col)])  # row, col
        sample = {'input': img, 'position': img_pos}

        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor:
    def __call__(self, sample):
        sample['input'] = VF.to_tensor(sample['input'])
        return sample


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                     transform=Compose([
                                         ToTensor()
                                     ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)


def test(args, bags_list, milnet):
    milnet.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    df = pd.read_csv('/app/Research/TransMIL/dataset_csv/camelyon16/SimCLRModelV0/fold0.csv')
    label_d = df.set_index('train')['train_label'].to_dict()
    label_d.update(df.set_index('val')['val_label'].to_dict())
    label_d.update(df.set_index('test')['test_label'].to_dict())
    del df
    for i in range(0, num_bags):
        # if i > 0:
        #     print(f'Bag_{i - 1} Time: {time.time() - start_time}')
        start_time = time.time()

        feats_list = []
        pos_list = []
        classes_list = []

        csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg')) + \
                        glob.glob(os.path.join(bags_list[i], '*.jpeg'))
        dataloader, bag_size = bag_dataset(args, csv_file_path)

        label_dict = {"0_normal": 0, "1_tumor": 1}
        label = None
        for key, value in label_dict.items():
            if key in bags_list[i]:
                label = value

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        label = label_d[bags_list[i].split('/')[-1]]
        print(f'{bags_list[i]} label: {label}')
        if label == 0:
            print(f'skipped {bags_list[i]} beacause it is Normal')
            continue

        slide_name = bags_list[i].split(os.sep)[-1]
        slide_output_path = os.path.join('test-c16', 'output', slide_name)
        #########################################
        # slide_input_path = os.path.join('test-c16', 'input', slide_name)
        level = 6  # level of tif file, max 9 for WSI and 8 for mask
        level_mask = 6
        alpha = 0.4  # alpha for blending
        get_mask = args.get_mask
        if get_mask:
            slide_input_path = os.path.join('test-c16', 'mask', slide_name + '_mask.tif')
        else:
            slide_input_path_mask = os.path.join('test-c16', 'mask', slide_name + '_mask.tif')
            slide_input_path = os.path.join('test-c16', 'input', slide_name + '.tif')
        dpi = 10
        cmap = 'gist_gray'
        #########################################
        os.makedirs(slide_output_path, exist_ok=True)


        m1 = m0 = -float('inf')
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i + 1, num_bags, iteration + 1, len(dataloader)))
                patches = batch['input'].float().cuda()
                patch_pos = batch['position']

                m0 = max(m0, np.amax(patch_pos.cpu().numpy(), 0)[0])
                m1 = max(m1, np.amax(patch_pos.cpu().numpy(), 0)[1])

                feats, classes = milnet.i_classifier(patches)
                feats = feats.cpu().numpy()
                classes = classes.cpu().numpy()

                feats_list.extend(feats)
                pos_list.extend(patch_pos)
                classes_list.extend(classes)

            pos_arr = np.vstack(pos_list)
            feats_arr = np.vstack(feats_list)
            classes_arr = np.vstack(classes_list)

            bag_feats = torch.from_numpy(feats_arr).cuda()
            ins_classes = torch.from_numpy(classes_arr).cuda()
            bag_feats = torch.from_numpy(feats_arr).unsqueeze(0).cuda()
            ins_classes = torch.from_numpy(classes_arr).unsqueeze(0).cuda()

            bag_prediction, A = milnet.b_classifier(bag_feats, ins_classes)
            bag_prediction = torch.sigmoid(bag_prediction).squeeze().cpu().numpy()
            attentions = A

            if bag_prediction >= args.thres_tumor:
                print(f'{bags_list[i]} is detected as malignant {1} ({bag_prediction}) | label: {label}')
                color = [1, 0, 0]
            else:
                print(f'{bags_list[i]} is detected as benign {0} ({bag_prediction}) | label: {label}')
                color = [0, 1, 0]

            attentions = ins_classes.squeeze()
            figure_path = os.path.join(slide_output_path, f'dsmil_inspred.png')
            #visualize(torch.sigmoid(ins_classes.squeeze()), pos_arr, color, figure_path)

            if not os.path.exists(slide_input_path):
                print(f'could not find: {slide_input_path}')
                continue

            # for WSI
            if get_mask:
                reader = mir.MultiResolutionImageReader()
                input_slide = reader.open(slide_input_path)
                input_image_size = input_slide.getLevelDimensions(level=level)
                x, y = input_slide.getLevelDimensions(level=0)
                input_slide = input_slide.getUCharPatch(startX=0, startY=0, width=input_image_size[0],
                                                        height=input_image_size[1], level=level)
            else:
                reader = mir.MultiResolutionImageReader()
                input_mask = reader.open(slide_input_path_mask)
                input_image_size = input_mask.getLevelDimensions(level=level_mask)
                x, y = input_mask.getLevelDimensions(level=0)
                input_mask = input_mask.getUCharPatch(startX=0, startY=0, width=input_image_size[0],
                                                      height=input_image_size[1], level=level_mask)
                input_slide = OpenSlide(slide_input_path)
                input_image_size = input_slide.level_dimensions[level]
                x, y = input_slide.level_dimensions[0]
                input_slide = input_slide.read_region((0, 0), level, input_image_size)

            slide_output_path = os.path.join(slide_output_path, 'cmaps')
            os.makedirs(slide_output_path, exist_ok=True)
            save_wsi = True
            for cmap in ['jet']:
                figure_path = os.path.join(slide_output_path, f'{cmap}')
                blend_and_visualize(ins_classes.squeeze(), pos_arr, color, figure_path, input_slide, alpha, x, y,
                                    input_image_size, dpi, get_mask, input_mask, cmap=cmap, save_wsi=save_wsi)

def blend_and_visualize(attentions, pos_arr, color, figure_path, input_image, alpha, x, y, input_img_size, dpi, is_mask,
                        mask, cmap='hot', invert=False, save_wsi=False):
    if invert:
        attentions = 1 - attentions
    # print('figure_path:', figure_path)
    # calculate the locations of attentions on input image
    xp = np.amax(pos_arr, 0)[0] + 1
    yp = np.amax(pos_arr, 0)[1] + 1
    tx = int(xp * 1024 * (input_img_size[1] / y))
    ty = int(yp * 1024 * (input_img_size[0] / x))
    tx = min(tx, input_img_size[1])
    ty = min(ty, input_img_size[0])
    #print(xp, yp, tx, ty, input_img_size[1], input_img_size[0])
    color_map = np.zeros((np.amax(pos_arr, 0)[0] + 1, np.amax(pos_arr, 0)[1] + 1))
    attentions = attentions.cpu().numpy()
    attentions = exposure.rescale_intensity(attentions, out_range=(0, 255))
    for k, pos in enumerate(pos_arr):
        color_map[pos[0], pos[1]] = attentions[k]
    color_map_size = color_map.shape
    color_map = transform.resize(color_map, (tx, ty), order=0)
    color_map_ = np.zeros((input_img_size[1], input_img_size[0]))
    #print('**************************************', color_map_.shape, color_map.shape)
    color_map_[:color_map.shape[0], :color_map.shape[1]] = color_map
    color_map = color_map_

    # prepare figure
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(input_img_size[1] // dpi * 6, input_img_size[0] // dpi * 6)
    fig.set_dpi(dpi)
    plt.axis('off')

    # save input image
    # input_image = np.array(input_image)
    # ax.imshow(input_image, cmap='gray')

    # input_image = np.array(input_image)
    ax.imshow(input_image.convert('L'), cmap='gray', alpha=0.7)

    # save heatmap
    color_map[color_map == 0] = np.nan
    ax.imshow(color_map, cmap=cmap, interpolation='none', alpha=alpha)

    # prepare and save mask for tumor slides
    mask = np.where(mask == 2, 1, 0)
    mask = Image.fromarray((mask * 255).astype(np.uint8).squeeze(2))
    mask = mask.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(size=11))
    mask = np.array(mask)
    mask = transform.resize(mask, (input_img_size[1], input_img_size[0]), order=0)
    mask_ = np.zeros((mask.shape[0], mask.shape[1], 4))
    mask_[:, :, 3] = (mask != 0)
    ax.imshow(mask_, interpolation='none')

    # save and close figure
    f = figure_path + '.png'
    fig.savefig(f, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close(fig)
    print(f'saved: {f}')

    # saving the WSI
    if save_wsi:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(input_img_size[1] // dpi * 6, input_img_size[0] // dpi * 6)
        fig.set_dpi(dpi)
        plt.axis('off')
        input_image = np.array(input_image)
        ax.imshow(input_image)
        figure_path = figure_path + '_slide.png'
        fig.savefig(figure_path, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close(fig)
        print(f'saved: {figure_path}')


def blend_and_visualize___(attentions, pos_arr, color, figure_path, input_image, alpha, x, y, input_img_size, dpi,
                           is_mask, cmap='hot'):
    xp = np.amax(pos_arr, 0)[0] + 1
    yp = np.amax(pos_arr, 0)[1] + 1
    tx = int(xp * 1024 * (input_img_size[1] / y))
    ty = int(yp * 1024 * (input_img_size[0] / x))
    color_map = np.zeros((np.amax(pos_arr, 0)[0] + 1, np.amax(pos_arr, 0)[1] + 1))
    attentions = attentions.cpu().numpy()
    attentions = exposure.rescale_intensity(attentions, out_range=(0, 255))
    for k, pos in enumerate(pos_arr):
        color_map[pos[0], pos[1]] = attentions[k]
    color_map_size = color_map.shape
    color_map = transform.resize(color_map, (tx, ty), order=0)
    color_map_ = np.zeros((input_img_size[1], input_img_size[0]))
    color_map_[:color_map.shape[0], :color_map.shape[1]] = color_map
    color_map = color_map_
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(input_img_size[1] // dpi * 6, input_img_size[0] // dpi * 6)
    fig.set_dpi(dpi)
    plt.axis('off')

    if not is_mask:
        input_image = input_image.convert('RGB')
        input_image = np.array(input_image)
        ax.imshow(input_image, alpha=0.65)
        color_map[color_map == 0] = np.nan
        heat_map = ax.imshow(color_map, cmap=cmap, interpolation='none', alpha=alpha)
    else:
        ax.imshow(input_image, cmap='gray')
        figure_path = figure_path[:-4] + '_mask.png'

    fig.savefig(figure_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close(fig)
    print(f'saved: {figure_path}')


def visualize(attentions, pos_arr, color, figure_path):
    # TODO make this faster by replacing the for loop with multiplication
    color_map = np.zeros((np.amax(pos_arr, 0)[0] + 1, np.amax(pos_arr, 0)[1] + 1, 3))
    attentions = attentions.cpu().numpy()
    attentions = exposure.rescale_intensity(attentions, out_range=(0, 1))
    for k, pos in enumerate(pos_arr):
        tile_color = np.asarray(color) * attentions[k]
        color_map[pos[0], pos[1]] = tile_color
    color_map = transform.resize(color_map, (color_map.shape[0] * 32, color_map.shape[1] * 32), order=0)
    io.imsave(figure_path, img_as_ubyte(color_map))


def blend_and_visualize_(attentions, pos_arr, color, figure_path, input_image, alpha, x, y, input_img_size, dpi):
    xp = np.amax(pos_arr, 0)[0] + 1
    yp = np.amax(pos_arr, 0)[1] + 1
    tx = int(xp * 1024 * (input_img_size[1] / y))
    ty = int(yp * 1024 * (input_img_size[0] / x))
    color_map = np.zeros((np.amax(pos_arr, 0)[0] + 1, np.amax(pos_arr, 0)[1] + 1, 3))
    attentions = attentions.cpu().numpy()
    attentions = exposure.rescale_intensity(attentions, out_range=(0, 1))
    for k, pos in enumerate(pos_arr):
        tile_color = np.asarray(color) * attentions[k]
        color_map[pos[0], pos[1]] = tile_color
    color_map_size = color_map.shape
    color_map = transform.resize(color_map, (tx, ty), order=0)
    color_map = img_as_ubyte(color_map)
    cmap = cm.get_cmap('RdBu_r')
    color_map = cmap(color_map).astype(np.uint8)
    color_map_ = np.zeros((input_img_size[1], input_img_size[0], 3))
    color_map_[:color_map.shape[0], :color_map.shape[1], :] = color_map
    color_map = Image.fromarray(color_map_.astype('uint8'), 'RGB')
    #
    # if input_image.mode != color_map.mode:
    #    input_image = input_image.convert(color_map.mode)
    # Image.blend(input_image, color_map, alpha).save(figure_path)
    #
    color_map = np.array(color_map)
    input_image = input_image.convert('RGB')
    input_image = np.array(input_image)
    input_image = np.where(color_map_ == 0, 0, input_image)
    fig, ax = plt.subplots(1, 1)

    ax.imshow(input_image)
    ax.imshow(color_map_, cmap='hot', alpha=0.5, cmin=1)

    plt.colorbar()

    plt.savefig(figure_path)
    plt.show()

    # heatmap = px.imshow(color_map, color_continuous_scale='hot', aspect="auto")

    # Overlay the heatmap on the image
    # fig = px.imshow(input_image, color_continuous_scale='gray')
    # fig.add_trace(heatmap)
    # fig.savefig(figure_path)

    # Close the figure
    # plt.close(fig)


def blend_and_visualize__(attentions, pos_arr, color, figure_path, input_image, alpha, x, y, input_img_size, dpi):
    # dpi = input_image.info['dpi']
    xp = np.amax(pos_arr, 0)[0] + 1
    yp = np.amax(pos_arr, 0)[1] + 1
    tx = int(xp * 1024 * (input_img_size[1] / y))
    ty = int(yp * 1024 * (input_img_size[0] / x))
    # print('------------------------------------------')
    # print(xp, yp, tx, ty, input_img_size[1], input_img_size[0], y, x)
    # print('------------------------------------------')
    color_map = np.zeros((np.amax(pos_arr, 0)[0] + 1, np.amax(pos_arr, 0)[1] + 1, 3))
    attentions = attentions.cpu().numpy()
    attentions = exposure.rescale_intensity(attentions, out_range=(0, 1))
    for k, pos in enumerate(pos_arr):
        tile_color = np.asarray(color) * attentions[k]
        color_map[pos[0], pos[1]] = tile_color
    color_map_size = color_map.shape
    color_map = transform.resize(color_map, (tx, ty), order=0)
    color_map = img_as_ubyte(color_map)
    color_map_ = np.zeros((input_img_size[1], input_img_size[0], 3))
    color_map_[:color_map.shape[0], :color_map.shape[1], :] = color_map
    color_map = np.transpose(color_map_, (1, 0, 2))
    color_map = Image.fromarray(color_map_.astype('uint8'), 'RGB')
    color_map = color_map.resize(input_image.size)
    color_map = np.array(color_map)
    # if input_image.mode != color_map.mode:
    #    input_image = input_image.convert(color_map.mode)
    # print('=============================================')
    # print(input_image.size, color_map.size)
    # print('=============================================')
    # Image.blend(input_image, color_map, alpha).save(figure_path)

    # color_map = np.transpose(color_map_)
    # color_map = Image.fromarray(color_map_.astype('uint8'), 'RGB')
    # color_map = color_map.resize(input_image.size)
    # color_map = np.array(color_map)
    # color_map = color_map_

    input_image = input_image.convert('RGB')
    input_image = np.array(input_image)
    # input_image = np.where(color_map_ == 0, 0, input_image)
    # print(input_image)
    # print(color_map_)
    fig, ax = plt.subplots(1, 1)

    ax.imshow(input_image)
    heat_map = ax.imshow(color_map, cmap='hot', alpha=0.5)

    # fig.colorbar(heat_map)
    # fig.set_size_inches(input_img_size[1] // dpi, input_img_size[0] // dpi)
    # print(input_img_size[1], input_img_size[0])
    # fig.set_dpi(dpi)
    plt.axis('off')
    fig.savefig(figure_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close(fig)


def save_colorbar(cmap, path):
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])

    cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=cmap)

    plt.savefig(path + '_colorbar.png', bbox_inches='tight')


def _get_embedder_official_weights(args, milnet: dsmil.MILNet):
    state_dict_weights = torch.load(args.embedder_weights)
    new_state_dict = OrderedDict()
    for i in range(4):
        state_dict_weights.popitem()
    state_dict_init = milnet.i_classifier.state_dict()
    for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
        name = k_0
        new_state_dict[name] = v
    return new_state_dict


def _load_embedder_weights(args, milnet: dsmil.MILNet):
    embedder_state_dict_weights = torch.load(args.embedder_weights)
    # embedder_state_dict_weights = _get_embedder_official_weights(args, milnet)

    # embedder_state_dict_weights = _remove_running_stats(embedder_state_dict_weights)  # For BatchNorm2d
    check_layers(milnet.i_classifier.state_dict(), embedder_state_dict_weights, header='Embedder')
    milnet.i_classifier.load_state_dict(embedder_state_dict_weights, strict=False)


def _load_aggregator_weights(args, milnet: dsmil.MILNet):
    aggregator_state_dict_weights = torch.load(args.aggregator_weights)
    aggregator_state_dict_weights["i_classifier.fc.weight"] = aggregator_state_dict_weights.pop(
        "i_classifier.fc.0.weight"
    )
    aggregator_state_dict_weights["i_classifier.fc.bias"] = aggregator_state_dict_weights.pop(
        "i_classifier.fc.0.bias"
    )
    check_layers(milnet.state_dict(), aggregator_state_dict_weights, header='Aggregator')
    milnet.load_state_dict(aggregator_state_dict_weights, strict=False)


def get_dsmil_milnet(args):
    resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()

    i_classifier = dsmil.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()
    b_classifier = dsmil.BClassifier(input_size=args.feats_size, output_class=args.num_classes).cuda()
    milnet = dsmil.MILNet(i_classifier, b_classifier).cuda()

    return milnet


def get_topk_milnet(args):
    resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()

    i_classifier = topk_dsmil.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()
    b_classifier = topk_dsmil.BClassifier(input_size=args.feats_size, output_class=1, dropout_v=0, nonlinear=1,
                                          num_heads=3,
                                          hidden_size=128, k=200).cuda()
    milnet = topk_dsmil.MILNet(i_classifier, b_classifier).cuda()
    return milnet


def get_milformer_milnet(args):
    resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()

    i_classifier = milformer.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()

    c = copy.deepcopy
    attn = milformer.MultiHeadedAttention(
        args.num_heads, args.feats_size, args.use_softmax_one
    ).cuda()
    ff = milformer.PositionwiseFeedForward(
        args.feats_size, args.feats_size * args.mlp_multiplier, args.encoder_dropout
    ).cuda()
    b_classifier = milformer.BClassifier(
        milformer.Encoder(
            milformer.EncoderLayer(
                args.feats_size, c(attn), c(ff), args.encoder_dropout, args.k, args.random_patch_share
            ), 1), args.num_classes, 512
    ).cuda()
    milnet = milformer.MILNet(i_classifier, b_classifier).cuda()

    _load_embedder_weights(args, milnet)
    _load_aggregator_weights(args, milnet)

    return milnet


def load_split_data(args, split_path, split_name):
    print(f'Loading {split_name} data... (mp={1})...')
    start_time = time.time()
    data = load_data(split_path, args)
    print(f'DONE (Took {(time.time() - start_time):.1f}s)')
    return data


def get_data(args, dataset='camelyon16', embedding='OfficialSimCLREmbedder'):
    """
    bag_df:         [column_0]                  [column_1]
                    path_to_bag_feats_csv       label
    """
    path_prefix = os.path.join('embeddings/', dataset, embedding)

    bags_csv = os.path.join(path_prefix, dataset + '.csv')
    bags_df = pd.read_csv(bags_csv)
    test_df = bags_df[bags_df['0'].str.startswith(f'{path_prefix}/test')]

    test_df = shuffle(test_df).reset_index(drop=True)

    test_data = load_split_data(args, test_df, 'test')

    return test_data


def get_dimentionality_reduction_data(args, milnet):
    milnet.eval()
    # num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    dataset = get_data(args)
    ys = []
    x_means = []
    with torch.no_grad():
        for y, feats in zip(dataset[0], dataset[1]):
            feats = Tensor(feats).cuda()
            feats = feats.view(feats.shape[0], -1)
            classes = milnet.i_classifier.fc(feats)
            t = milnet.b_classifier.encoder(feats.unsqueeze(dim=0), classes.unsqueeze(dim=0))[0].mean(dim=1)
            x_means.append(t.cpu().numpy())
            ys.append(y)

    x_means = np.vstack(x_means)
    y = np.vstack(ys).squeeze()

    return x_means, y


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Testing workflow includes attention computing and color map production'
    )
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--n_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of feeding patches')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--feats_size', type=int, default=512)
    parser.add_argument('--thres_tumor', type=float, default=0.1964)
    parser.add_argument('--embedder_weights', type=str, default=os.path.join('test-c16', 'weights', 'embedder.pth'))
    parser.add_argument('--aggregator_weights', type=str, default=os.path.join('test-c16', 'weights', 'aggregator.pth'))
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--use_softmax_one', default=1, type=int, help='using the modified type of softmax or not')
    parser.add_argument('--mlp_multiplier', default=4, type=int, help='inverted mlp anti-bottbleneck')
    parser.add_argument('--encoder_dropout', default=0.0, type=float, help='dropout in encoder')
    parser.add_argument('--k', default=200, type=int, help='top k')
    parser.add_argument('--random_patch_share', default=0.0, type=float, help='dropout in encoder')
    parser.add_argument('--t_SNE', default=False, type=bool, help='calculate t-SNE')
    parser.add_argument('--PCA', default=False, type=bool, help='calculate PCA')
    parser.add_argument('--use_mp', default=True, type=bool, help='multiproccess')
    parser.add_argument('--num_processes', default=None, type=int, help='number of processes')
    parser.add_argument('--perplexity', default=30, type=int, help='preplexity of t-SNE algorithm')
    parser.add_argument('--UMAP', default=False, type=bool, help='calculate umap')
    parser.add_argument('--min_dist', default=0.1, type=int, help='min dist or umap')
    parser.add_argument('--n_neighbors', default=10, type=int, help='number of neighbors for umap')
    parser.add_argument('--get_mask', default=False, type=bool, help='visualize mask or WSI')
    args = parser.parse_args()

    # milnet = get_dsmil_milnet(args)
    # milnet = get_topk_milnet(args)
    milnet = get_milformer_milnet(args)

    bags_path = os.path.join('test-c16', 'patches', '*')
    bags_list = glob.glob(bags_path)
    print(f'len(bags_list): {len(bags_list)} | bags_list[0]: {bags_list[0]}')
    os.makedirs(os.path.join('test-c16', 'output'), exist_ok=True)
    test(args, bags_list, milnet)