from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        #  Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--tune_steps', type=int, default=10, help='tune_steps')
        parser.add_argument('--tune_lr', type=float, default=0.01, help='tune_lr')
        parser.add_argument('--test_config', required=False, is_config_file=True, help='test config file path')

        parser.add_argument('--used_points', type=str, required=True, help='all|original')
        parser.add_argument('--error_form', type=str, required=True, help='all|image_size')
        parser.add_argument('--num_test_save', type=int, default=30, help='')
        parser.add_argument('--print_freq', type=int, default=5, help='')

        parser.set_defaults(subset='test')
        parser.set_defaults(model='test')
        parser.set_defaults(allow_unknown_options=True)
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        self.isTrain = False
        return parser
