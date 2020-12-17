import cPickle as cp
import itertools
import os
import pickle
from argparse import Namespace

import torch
from torch.autograd import Variable
from torch.nn import functional as F

import data.human36m_skeleton
from data import CreateDataLoader
from data.human36m_skeleton import simple_link_indices as human36m_link_indices
from util import plotting, util
from util.image_pool import ImagePool
from util.tps_sampler import TPSRandomSampler

from . import networks, utils
from .base_model import BaseModel
from .perceptual_loss import PerceptualLoss


class KeypointGANModel(BaseModel):
    def name(self):
        return 'KeypointGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_gan_A', type=float, default=1.0, help='weight for gan loss')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # opt.phase in ['train', 'train_regressor']
        self.mode = opt.phase
        self.no_grad = opt.phase == 'test'

        # ----------------------- Losses to print-------------------------------
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        if opt.phase == 'train':
            self.loss_names = []

            if not opt.not_optimize_G:
                self.loss_names += ['cycle_A']
                self.loss_names += ['G_A']
                if opt.lambda_render_consistency > 0:
                    self.loss_names += ['render_consistency']

            if not opt.not_optimize_D:
                self.loss_names += ['D_A']

            if opt.finetune_regressor:
                self.loss_names += ['regressor']

        elif opt.phase == 'train_regressor':
            self.loss_names = ['regressor']

        # ----------------------- Visualizations -------------------------------
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if opt.phase in ['train', 'test']:
            if opt.eval_pose_prediction_only:
                visual_names_A = ['real_A',
                                  'fake_B', 'fake_B_regress']
            else:
                visual_names_A = ['real_A',
                                'fake_B', 'fake_B_regress', 'rec_A']
                if opt.phase != 'test':
                    visual_names_A.append('real_B')

                visual_names_A.insert(1, 'real_cond_A')

                if self.opt.offline_regressor:
                    visual_names_A += ['offline_regress']

                visual_names_B = []

                self.visual_names = visual_names_A + visual_names_B
        if opt.phase in ['train_regressor']:
            self.visual_names = ['real_B', 'fake_B_regress']

        # ----------------------- Networks save/load----------------------------
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        model_names = ['G_B']
        if self.isTrain:
            model_names += ['G_A']
            model_names += ['D_A']
        else:  # during test time, only load Gs
            model_names += ['G_A']

        if opt.phase == 'train_regressor':
            self.load_model_names = ['regressor']
            self.save_model_names = ['regressor']
        else:
            self.load_model_names = model_names
            self.load_model_names += ['regressor']
            if self.opt.offline_regressor:
                self.load_model_names += ['offline_regressor']
            self.save_model_names = model_names

        if opt.finetune_regressor:
            self.save_model_names += ['regressor']

        # ----------------------- Define networks ------------------------------
        # load/define networks
        self.netregressor = networks.define_regressor(
            1, self.opt.n_points, norm=opt.regressor_norm, init_type=opt.init_type,
            init_gain=opt.init_gain, gpu_ids=self.gpu_ids,
            net_type=opt.net_regressor, n_channels=opt.net_regressor_channels)

        if self.opt.offline_regressor:
            self.netoffline_regressor = networks.define_regressor(
                1, self.opt.n_points, norm=opt.regressor_norm, init_type=opt.init_type,
                init_gain=opt.init_gain, gpu_ids=self.gpu_ids,
                net_type=opt.net_regressor, n_channels=opt.net_regressor_channels)

        self.netG_A = networks.define_G(
            opt.input_nc, opt.output_nc, opt.netG_A, norm=opt.generators_norm,
            init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids, 
            n_blocks=opt.netG_A_blocks)

        netG_input_nc = opt.output_nc
        self.netG_B = networks.define_G_cond(
            netG_input_nc, opt.input_nc, opt.input_nc, opt.netG_B, norm=opt.generators_norm,
            init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids,
            avg_pool_cond=opt.avg_pool_style)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(
                opt.output_nc, opt.ndf, opt.netDA, multi_gan=opt.multi_ganA,
                n_layers_D=opt.n_layers_D, norm=opt.discriminators_norm, 
                use_sigmoid=use_sigmoid, init_type=opt.init_type, init_gain=opt.init_gain, 
                gpu_ids=self.gpu_ids)

        if self.isTrain:
            # ------------------------- Criterions -----------------------------
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            gan_loss = networks.MultiGANLoss if opt.multi_ganA else networks.GANLoss
            self.criterionGAN = gan_loss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.single_scale_criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            if self.opt.cycle_loss == 'l1':
                self.criterionCycle = torch.nn.L1Loss()
            elif self.opt.cycle_loss == 'perceptual':
                self.criterionCycle = PerceptualLoss(self.opt.perceptual_net)
                    # '/scratch/local/hdd/ankush/minmaxinfo/data/models/imagenet-vgg-verydeep-19.mat'
            else:
                raise ValueError('Unknown cycle loss: %s' % self.opt.cycle_loss)
            self.criterionIdt = torch.nn.L1Loss()
            self.criterion_regressor = torch.nn.MSELoss()

            # -------------------------- Optimizers ----------------------------
            # initialize optimizers
            G_params = [self.netG_B.parameters()]
            G_params += [self.netG_A.parameters()]
            self.optimizer_G = torch.optim.Adam(itertools.chain(*G_params),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            D_params = []
            D_params += [self.netD_A.parameters()]
            self.optimizer_D = torch.optim.Adam(itertools.chain(*D_params), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_regressor = torch.optim.Adam(
                self.netregressor.parameters(), lr=opt.lr,
                betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_regressor)

        # ------------------------------ TPS -----------------------------------
        if self.opt.tps:
            self.tps_sampler = TPSRandomSampler(
                opt.fineSize, opt.fineSize, rotsd=5.0, scalesd=0.05, transsd=0.05,
                warpsd=(0.0005, 0.005))
            self.tps_sampler_target = TPSRandomSampler(
                opt.fineSize, opt.fineSize, rotsd=5.0, scalesd=0.05, transsd=0.05,
                warpsd=(0.0, 0.0))


    def set_input(self, input):
        self.input = input

        if 'A' in input:
            self.real_A = input['A'].to(self.device)
        if 'cond_A' in input:
            self.real_cond_A = input['cond_A'].to(self.device)
        self.real_B_points = input['B'].to(self.device)

        if 'paired_cond_B' in input:
            self.paired_cond_B_points = input['paired_cond_B'].to(self.device)
            self.paired_cond_B = self.render_skeleton(
                self.paired_cond_B_points, skeleton_type=self.opt.paired_skeleton_type)

        if 'paired_B' in input:
            self.paired_B = input['paired_B'].to(self.device)
        else:
            self.paired_B = self.real_B_points

        if 'B_visible' in input:
            self.B_visible = input['B_visible'].to(self.device)
        if 'paired_B_visible' in input:
            self.paired_B_visible = input['paired_B_visible'].to(self.device)

        if 'A_paths' in input:
            self.image_paths = input['A_paths']
        # warp input
        if self.opt.tps:
            self.real_cond_A = self.tps_sampler(self.real_cond_A)
        if self.opt.tps_target:
            self.real_A = self.tps_sampler_target(self.real_A)

        # shuffle real_cond_A
        if self.opt.shuffle_identities:
            torch.manual_seed(0)
            self.real_cond_A = self.real_cond_A[torch.randperm(self.real_cond_A.shape[0])]

        self.paired_B_points = self.paired_B

        skeleton_type = self.opt.skeleton_type
        if self.opt.prior_skeleton_type is not None:
            skeleton_type = self.opt.prior_skeleton_type
        self.real_B = self.render_skeleton(
            self.real_B_points, skeleton_type=skeleton_type, reduce=self.opt.reduce_rendering_mode)

    def forward(self):
        if self.mode == 'train_regressor':
            maps = self.netregressor(self.real_B)
            self.regressed_points = utils.extract_points(maps)
            skeleton_type = self.opt.skeleton_type
            if self.opt.prior_skeleton_type is not None:
                skeleton_type = self.opt.prior_skeleton_type
            self.fake_B_regress = self.render_skeleton(
                self.regressed_points, skeleton_type=skeleton_type, 
                reduce='max')

        elif not self.opt.eval_pose_prediction_only:
            if self.opt.finetune_regressor:
                maps = self.netregressor(self.real_B)
                self.real_B_regressed_points = utils.extract_points(maps)
                self.real_B_regress = self.render_skeleton(
                    self.real_B_regressed_points,
                    skeleton_type=self.opt.skeleton_type)

            self.fake_B = self.netG_A(self.real_A)
            maps = self.netregressor(self.fake_B)

            self.regressed_points = utils.extract_points(maps)

            skeleton_type = self.opt.skeleton_type
            if self.opt.prior_skeleton_type is not None:
                skeleton_type = self.opt.prior_skeleton_type
            fake_B_regress_multi_ch = self.render_skeleton(
                self.regressed_points, reduce=None,
                skeleton_type=skeleton_type)
            self.fake_B_regress = self.reduce_renderings(
                    fake_B_regress_multi_ch, reduce=self.opt.reduce_rendering_mode, keepdim=True)
            
            if self.opt.offline_regressor:
                # offline_regressor_input
                maps = self.netoffline_regressor(self.fake_B_regress)
                self.offline_regressed_points = utils.extract_points(maps)
                self.offline_regress = self.render_skeleton(
                    self.offline_regressed_points, colored=True,
                    skeleton_type=self.opt.skeleton_type)

            netG_B_input = self.fake_B_regress

            self.rec_A = self.netG_B(netG_B_input, self.real_cond_A)


    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_single_scale(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.single_scale_criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.single_scale_criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D


    def backward_D_A(self):
        fake_B = self.fake_B
        real_B = self.real_B
        fake_B = self.fake_B_pool.query(fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, real_B, fake_B)


    def backward_G(self, retain_graph=False):
        lambda_A = self.opt.lambda_A
        lambda_gan_A = self.opt.lambda_gan_A

        # GAN loss D_A(G_A(A))
        fake_B = self.fake_B
        self.loss_G_A = self.criterionGAN(self.netD_A(fake_B), True) * lambda_gan_A

        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

        self.loss_render_consistency = 0
        if self.opt.lambda_render_consistency > 0:
            self.loss_render_consistency = self.criterion_regressor(
                self.fake_B, self.fake_B_regress.detach()) * self.opt.lambda_render_consistency

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_cycle_A + self.loss_render_consistency
        self.loss_G.backward(retain_graph=retain_graph)


    def backward_regressor(self):
        regressed_points = self.regressed_points
        real_B_points = self.real_B_points
        if self.opt.only_visible_points_loss:
            regressed_points = regressed_points * self.B_visible[:, :, None].type(regressed_points.dtype)
            real_B_points = real_B_points * self.B_visible[:, :, None].type(real_B_points.dtype)
        self.loss_regressor = self.criterion_regressor(regressed_points, real_B_points)
        if self.opt.regressor_im_loss > 0:
            self.loss_regressor += self.opt.regressor_im_loss * self.criterion_regressor(
                self.fake_B_regress, self.real_B)
        self.loss_regressor.backward()


    def backward_regressor_finetune(self):
        self.loss_regressor = self.opt.regressor_fake_loss * self.criterion_regressor(
                self.fake_B_regress, self.fake_B.detach())
        real_B_regressed_points = self.real_B_regressed_points
        real_B_points = self.real_B_points
        if self.opt.only_visible_points_loss:
            real_B_regressed_points = real_B_regressed_points * self.B_visible[:, :, None].type(real_B_regressed_points.dtype)
            real_B_points = real_B_points * self.B_visible[:, :, None].type(real_B_points.dtype)
        self.loss_regressor += self.opt.regressor_real_loss * self.criterion_regressor(
                real_B_regressed_points, real_B_points)
        self.loss_regressor.backward()


    def optimize_parameters(self):
        # forward
        all_nets_but_regressor = [self.netG_B]
        all_nets_but_regressor += [self.netG_A]
        all_nets_but_regressor += [self.netD_A]

        self.forward()

        if self.mode == 'train_regressor':
            self.set_requires_grad(all_nets_but_regressor, False)
            self.optimizer_regressor.zero_grad()
            self.backward_regressor()
            if self.opt.clip_grad < float('inf'):
                self.clip_gradient(self.optimizer_regressor, self.opt.clip_grad)
            self.optimizer_regressor.step()

        else:
            D = []
            D += [self.netD_A]

            self.set_requires_grad(self.netregressor, False)
            retain_graph = self.opt.finetune_regressor

            # G_A and G_B
            if not self.opt.not_optimize_G:
                self.set_requires_grad(D, False)
                self.optimizer_G.zero_grad()
                self.backward_G(retain_graph=retain_graph)
                if self.opt.clip_grad < float('inf'):
                    self.clip_gradient(self.optimizer_G, self.opt.clip_grad)
                self.optimizer_G.step()

            # D_A
            if not self.opt.not_optimize_D and len(D) > 0:
                self.set_requires_grad(D, True)
                self.optimizer_D.zero_grad()
                self.backward_D_A()
                if self.opt.clip_grad < float('inf'):
                    self.clip_gradient(self.optimizer_D, self.opt.clip_grad)
                self.optimizer_D.step()

            # finetune regressor
            if self.opt.finetune_regressor:
                self.set_requires_grad(all_nets_but_regressor, False)
                self.set_requires_grad(self.netregressor, True)
                self.optimizer_regressor.zero_grad()
                self.backward_regressor_finetune()
                if self.opt.clip_grad < float('inf'):
                    self.clip_gradient(self.optimizer_regressor, self.opt.clip_grad)
                self.optimizer_regressor.step()
                self.set_requires_grad(all_nets_but_regressor, True)



    def reduce_renderings(self, render, reduce='max', keepdim=True):
        if reduce == 'softmax':
            weights = F.softmax(render, dim=1)
            render = torch.sum(render * weights, dim=1, keepdim=keepdim)
        elif reduce == 'mean':
            render = torch.mean(render, dim=1, keepdim=keepdim)
        elif reduce == 'sum':
            render = torch.sum(render, dim=1, keepdim=keepdim)
        elif reduce == 'max':
            render, _ = torch.max(render, dim=1, keepdim=keepdim)
        elif reduce is None:
            pass
        else:
            ValueError()
        return render


    def get_link_indices(self, skeleton_type):
        if skeleton_type == 'human36m':
            link_indices = human36m_link_indices
        elif skeleton_type == 'human36m_simple2':
            link_indices = data.human36m_skeleton.simple2_link_indices
        elif skeleton_type == 'disconnected':
            link_indices = self.get_disconnected_links(self.opt.n_points)
        else:
            raise ValueError()
        return link_indices


    def render_skeleton(self, points, reduce='max', colored=None,
                        skeleton_type='human36m', colors=None, centre=True,
                        normalize=False, widths=None, size=None):
        if size is None:
            size = (self.opt.fineSize, self.opt.fineSize)

        if skeleton_type != 'points':
            link_indices = self.get_link_indices(skeleton_type)
            render_fn = utils.render_skeleton
            render = render_fn(
                points, link_indices,
                size[0], size[1], colored=colored,
                colors=colors,
                normalize=normalize, widths=widths,
                sigma=self.opt.sigma)
        elif skeleton_type == 'points':
            render = utils.render_points(
                points, size[0], size[1])
            render = render[:, :, None]
        else:
            raise ValueError()

        render = self.reduce_renderings(render, reduce=reduce)

        if colors is None:
            render = torch.mean(render, dim=2)

        if centre:
            render = utils.normalize_im(render)

        return render


    def get_limb_points(self, points, skeleton_type):
        connections = self.get_link_indices(skeleton_type)
        return utils.get_line_points(points, connections)


    def clip_gradient(self, optimizer, clip):
        for param_group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(param_group['params'], clip)


    def compute_visuals(self):
        if 'train' in self.opt.phase:
            return

        self.visual_names.append('real_A_paired_B')
        self.visual_names.append('real_A_fake_B')

        if self.opt.plot_skeleton_type is not None:
            skeleton_type = self.opt.plot_skeleton_type
        else:
            skeleton_type = self.opt.skeleton_type

        if skeleton_type != 'points':
            link_indices = self.get_link_indices(skeleton_type)

        points = self.regressed_points
        real_points = self.paired_B_points

        if self.opt.offline_regressor:
            corrected_points = self.correct_flips(points, self.offline_regressed_points)

        if skeleton_type != 'points':
            points, new_link_indices = utils.parse_auxiliary_links(points, link_indices)
            real_points, _ = utils.parse_auxiliary_links(real_points, link_indices)

        if self.opt.offline_regressor:
            self.visual_names.append('real_A_fake_B_offline')
            offline_points = self.offline_regressed_points
            offline_points, _ = utils.parse_auxiliary_links(offline_points, link_indices)

            self.visual_names.append('real_A_fake_B_offline_adj')
            corrected_points, _ = utils.parse_auxiliary_links(corrected_points, link_indices)
        
        if skeleton_type != 'points':    
            link_indices = new_link_indices

        if skeleton_type != 'points':
            self.real_A_paired_B = plotting.plot_in_image(
                util.tensor2im(self.real_A), real_points[0].cpu().numpy(),
                color='navy', style='skeleton', connections=link_indices)
            self.real_A_fake_B = plotting.plot_in_image(
                util.tensor2im(
                    self.real_A), points[0].cpu().numpy(),
                color='limegreen', style='skeleton', connections=link_indices)
            if self.opt.offline_regressor:
                self.real_A_fake_B_offline = plotting.plot_in_image(
                    util.tensor2im(self.real_A), offline_points[0].cpu().numpy(),
                    color='navy', style='skeleton', connections=link_indices)
                self.real_A_fake_B_offline_adj = plotting.plot_in_image(
                    util.tensor2im(self.real_A), corrected_points[0].cpu().numpy(),
                    color='navy', style='skeleton', connections=link_indices)
        elif skeleton_type == 'points':
            self.real_A_paired_B = plotting.plot_in_image(
                util.tensor2im(self.real_A), real_points[0].cpu().numpy(),
                color='cyan', landmark_size=self.opt.plot_landmark_size)
            self.real_A_fake_B = plotting.plot_in_image(
                util.tensor2im(
                    self.real_A), points[0].cpu().numpy(),
                    color='limegreen', landmark_size=self.opt.plot_landmark_size)
        else:
            raise ValueError()


    def get_disconnected_links(self, n_points):
        return [(i, i + 1) for i in range(0, n_points, 2)]


    def normalize_points(self, landmarks):
        # put in the corner
        minv, _ = torch.min(landmarks, dim=1, keepdim=True)
        landmarks = landmarks - minv
        # normalize between -1, 1
        height_width, _ = torch.max(landmarks, dim=1, keepdim=True)
        size, _ = torch.max(height_width, dim=2, keepdim=True)
        landmarks = 2.0 * (landmarks / size) - 1.0
        # centre
        maxv, _ = torch.max(landmarks, dim=1, keepdim=True)
        landmarks = landmarks + (1.0 - maxv) / 2.0
        return landmarks


    def correct_flips(self, input, offline_prediction):
        if self.opt.skeleton_type in ['human36m', 'human36m_simple2']:
            correspondences = data.human36m_skeleton.get_lr_correspondences()
        else:
            raise ValueError()

        input_swapped = utils.swap_points(input, correspondences)

        distance = utils.mean_l2_distance(offline_prediction, input)
        swapped_distance = utils.mean_l2_distance(offline_prediction, input_swapped)
        min_idx = distance > swapped_distance
        corrected_input = torch.zeros_like(input)
        for i in range(len(min_idx)):
            if min_idx[i]:
                corrected_input[i] = input_swapped[i]
            else:
                corrected_input[i] = input[i]
        return corrected_input
