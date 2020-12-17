import configargparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')
        parser.add_argument('--dataroot', required=False, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--netDA', type=str, default='basic', help='discriminator for fwd arm')
        parser.add_argument('--netG_A', type=str, default='resnet_9blocks', help='selects model to use for netG_A')
        parser.add_argument('--netG_B', type=str, default='nips', help='selects model to use for netG_B')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses dataset')
        parser.add_argument('--model', type=str, default='keypoint_gan', help='chooses which model to use')
        parser.add_argument('--iteration', type=str, default='latest', help='which iter to load? set to latest to use latest cached model')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        parser.add_argument('--tps', action='store_true', help='random TPS on input')
        parser.add_argument('--tps_target', action='store_true', help='random TPS on input')
        parser.add_argument('--multi_ganA', action='store_true', help='mulit GAN')
        parser.add_argument('--perceptual_net', type=str, default='', help='path to perceptual net')
        parser.add_argument('--nets_paths', type=str, nargs='+', help='list of pairs <net name> <net path> inits nets from the specified checkpoints')
        parser.add_argument('--upsampling_G_A', type=str, default='transpose', help='')
        parser.add_argument('--skeleton_type', type=str, default='human36m', help='')
        parser.add_argument('--paired_skeleton_type', type=str, default='human36m', help='')
        parser.add_argument('--n_points', type=int, default='32', help='')
        parser.add_argument('--subset', type=str, default='train', help='')
        parser.add_argument('--allow_unknown_options', action='store_true', help='')
        parser.add_argument('--shuffle', type=str, default='false', help='')
        parser.add_argument('--shuffle_identities', action='store_true', help='')
        parser.add_argument('--regressor_norm', type=str, default='instance', help='')
        parser.add_argument('--discriminators_norm', type=str, default='instance', help='')
        parser.add_argument('--generators_norm', type=str, default='batch', help='')
        parser.add_argument('--regressor_im_loss', type=float, default=0, help='')
        parser.add_argument('--finetune_regressor', action='store_true', help='')
        parser.add_argument('--reduce_rendering_mode', type=str, default='max', help='')
        parser.add_argument('--net_regressor', type=str, default='nips_encoder', help='')
        parser.add_argument('--net_regressor_channels', type=int, default=32, help='')
        parser.add_argument('--offline_regressor', action='store_true', help='')
        parser.add_argument('--eval_pose_prediction_only', action='store_true', help='')
        parser.add_argument('--prior_skeleton_type', type=str, default=None, help='')
        parser.add_argument('--plot_skeleton_type', type=str, default=None, help='')
        parser.add_argument('--sigma', type=float, default=0.4, help='')
        parser.add_argument('--avg_pool_style', action='store_true', help='')
        parser.add_argument('--netG_A_blocks', type=int, default=4, help='')
        parser.add_argument('--source_tps_params', type=float, default=[5.0, 0.05, 0.05, 0.0005, 0.005], nargs=5, help='')
        parser.add_argument('--target_tps_params', type=float, default=[5.0, 0.05, 0.05, 0.0, 0.0], nargs=5, help='')
        parser.add_argument('--plot_landmark_size', type=float, default=1.3, help='')
        parser.add_argument('--resume_from_name', type=str, default=None, help='')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = configargparse.ArgumentParser(
                formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser

        if hasattr(opt, 'allow_unknown_options') and opt.allow_unknown_options:
            opt, unknown = parser.parse_known_args()
        else:
            opt = parser.parse_args()
            unknown = []

        return opt, unknown

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def print_unknown(self, unknown):
        message = ''
        message += '----------------- Unknown options ---------------\n'
        for item in unknown:
            if item.startswith('-'):
                message += '%s, ' % item
        message += '\n'
        message += '----------------- End -------------------'
        print(message)

    def parse(self):

        opt, unknown = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)
        if unknown:
            self.print_unknown(unknown)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
