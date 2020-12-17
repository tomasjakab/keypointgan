from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=20000, help='frequency of saving the latest results')
        parser.add_argument('--save_iters_freq', type=int, default=20000, help='frequency of saving checkpoints')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--iters_count', type=int, default=1, help='the starting iters count, we save the model by <iters_count>, <iters_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # TODO: lr_policy fixed to none, needs to be adapted for iterations (was using epochs)
        parser.add_argument('--lr_policy', type=str, default='none', help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--cycle_loss', type=str, default='l1', help='cycle loss: l1|perceptual')
        parser.add_argument('--clip_grad', type=float, default=float('inf'), help='')
        parser.add_argument('--not_optimize_G', action='store_true', help='')
        parser.add_argument('--not_optimize_D', action='store_true', help='')
        parser.add_argument('--regressor_fake_loss', type=float, default=0.0, help='')
        parser.add_argument('--regressor_real_loss', type=float, default=0.0, help='')
        parser.add_argument('--lambda_render_consistency', type=float, default=0.0, help='')
        parser.add_argument('--only_visible_points_loss', action='store_true', help='')

        self.isTrain = True
        return parser
