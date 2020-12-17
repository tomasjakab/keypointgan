import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F



###############################################################################
# Helper Functions
###############################################################################
class Interpolate(nn.Module):
    def __init__(self, size=None, scale=None, mode='nearest'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.scale = scale
        self.mode = mode
        self.align_corners = None if mode == 'nearest' else False

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale,
                        mode=self.mode, align_corners=self.align_corners)
        return x


def get_norm_layer(norm_type='instance', dims=2):
    if norm_type == 'batch':
        if dims == 1:
            layer = nn.BatchNorm1d
        elif dims == 2:
            layer = nn.BatchNorm2d
        else:
            raise NotImplementedError('unsupported dim: %d' % dims)
        norm_layer = functools.partial(layer, affine=True)
    elif norm_type == 'instance':
        if dims == 1:
            layer = nn.InstanceNorm1d
        elif dims == 2:
            layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError('unsupported dim: %d' % dims)
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

# TODO: needs to be adapted to iterations
def get_scheduler(optimizer, opt):
    assert opt.lr_policy == 'none'
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    elif opt.lr_policy == 'none':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=1000000000000, gamma=1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, netG, norm='batch',
             init_type='normal', init_gain=0.02, gpu_ids=[], n_blocks=4):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'skip_nips':
        net = SkipNipsGenerator(input_nc, output_nc, n_blocks=n_blocks)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_G_cond(input_nc, cond_nc, output_nc, netG, norm='batch', 
                  init_type='normal', init_gain=0.02, gpu_ids=[], avg_pool_cond=False):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'nips':
        net = CondNipsGenerator(input_nc, cond_nc, output_nc, avg_pool_cond=avg_pool_cond)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, multi_gan=False,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    def net_factory(net_class, *args, **kwargs):
        return lambda: net_class(*args, **kwargs)

    if netD == 'basic':
        net = net_factory(
            NLayerDiscriminator, input_nc, ndf, n_layers=3, 
            norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    if multi_gan:
        net = MultiDiscriminator(net)
    else:
        net = net()
    return init_net(net, init_type, init_gain, gpu_ids)


def define_regressor(input_nc, output_nc, norm='batch', init_type='normal',
                     init_gain=0.02, gpu_ids=[], net_type='nips_encoder',
                     n_channels=32):
    norm_layer = get_norm_layer(norm_type=norm)
    if net_type == 'nips_encoder':
        net = NipsEncoder(input_nc, output_nc, norm_layer=norm_layer,
                          channels=n_channels)
    else:
        raise NotImplementedError('Regressor model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class MultiGANLoss(GANLoss):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(MultiGANLoss, self).__init__(
            use_lsgan=use_lsgan, target_real_label=target_real_label,
            target_fake_label=target_fake_label)

    def __call__(self, inputs, target_is_real):
        losses = []
        for input in inputs:
            loss = (super(MultiGANLoss, self).__call__(input, target_is_real))
            losses.append(loss)
        return torch.mean(torch.stack(losses))


class NormalizedLoss(nn.Module):
    def __init__(self, base_loss, mu=0.999, init_val=None):
        super(NormalizedLoss, self).__init__()
        self.add_module('base_loss', base_loss)
        # self.base_loss = base_loss
        self.mu = mu
        self.init_val = init_val
        self.register_buffer('running_mean', torch.tensor(init_val or -1.0))

    def __call__(self, *args, **kw_args):
        curr_loss = self.base_loss(*args, **kw_args)
        # update the moving average:
        if self.running_mean.device != curr_loss.device:
            self.running_mean = self.running_mean.to(curr_loss.device)
        if self.running_mean == -1.0:
            self.running_mean = curr_loss.detach()
        else:
            self.running_mean = self.mu * self.running_mean  + (1.0 - self.mu) * curr_loss.detach()
        loss_val = curr_loss / (self.running_mean + 1e-8)
        return loss_val



class SkipNipsGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, n_blocks=4, norm_layer=nn.BatchNorm2d):
        super(SkipNipsGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc

        decoder_inputs = [256, 128, 64, 32][4-n_blocks:]
        bottleneck_nc = decoder_inputs[0]

        self.encoder = SkipNipsEncoder(input_nc, bottleneck_nc, n_blocks=n_blocks, norm_layer=norm_layer)
        self.decoder = SkipNipsDecoder(decoder_inputs, output_nc, n_blocks=n_blocks, norm_layer=norm_layer)

    def forward(self, input):
        outputs = self.encoder(input)
        return self.decoder(outputs[::-1])



class CondNipsGenerator(nn.Module):
    def __init__(self, input_nc, cond_nc, output_nc, norm_layer=nn.BatchNorm2d, 
                 avg_pool_cond=False):
        super(CondNipsGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.avg_pool_cond = avg_pool_cond

        bottleneck_nc = 256

        self.encoder = NipsEncoder(input_nc, bottleneck_nc, norm_layer=norm_layer)
        self.encoder_cond = NipsEncoder(cond_nc, bottleneck_nc, norm_layer=norm_layer)
        self.decoder = NipsDecoder(2 * bottleneck_nc, output_nc, norm_layer=norm_layer)

    def forward(self, input, cond):
        input = self.encoder(input)
        cond = self.encoder_cond(cond)
        if self.avg_pool_cond:
            spatial_size = list(cond.shape[2:])
            cond = torch.mean(cond, dim=(2, 3), keepdim=True)
            cond = cond.repeat([1, 1] + spatial_size)
        decoder_input = torch.cat([input, cond], dim=1)
        output = self.decoder(decoder_input)
        return output



class NipsEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, channels=32,
                 norm_layer=nn.BatchNorm2d):
        super(NipsEncoder, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []

        def conv(channels_in, channels_out, kernel_size=3, stride=1, padding=1):
            return [nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=use_bias),
                    norm_layer(channels_out),
                    nn.ReLU(True)]

        channels = 32
        model += conv(input_nc, channels, kernel_size=7, stride=1, padding=3)
        model += conv(channels, channels, kernel_size=3, stride=1, padding=1)

        channels_in = channels
        channels *= 2
        model += conv(channels_in, channels, kernel_size=3, stride=2, padding=1)
        model += conv(channels, channels, kernel_size=3, stride=1, padding=1)

        channels_in = channels
        channels *= 2
        model += conv(channels_in, channels, kernel_size=3, stride=2, padding=1)
        model += conv(channels, channels, kernel_size=3, stride=1, padding=1)

        channels_in = channels
        channels *= 2
        model += conv(channels_in, channels, kernel_size=3, stride=2, padding=1)
        model += conv(channels, channels, kernel_size=3, stride=1, padding=1)

        model += [nn.Conv2d(channels, output_nc, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class SkipNipsEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, channels=32, n_blocks=4,
                 norm_layer=nn.BatchNorm2d):
        super(SkipNipsEncoder, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d


        def conv(channels_in, channels_out, kernel_size=3, stride=1, padding=1):
            return [nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=use_bias),
                    norm_layer(channels_out),
                    nn.ReLU(True)]

        self.n_blocks = n_blocks

        model = []
        channels = 32
        model += conv(input_nc, channels, kernel_size=7, stride=1, padding=3)
        model += conv(channels, channels, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.ModuleList([nn.Sequential(*model)])

        for _ in range(n_blocks - 2):
            model = []
            channels_in = channels
            channels *= 2
            model += conv(channels_in, channels, kernel_size=3, stride=2, padding=1)
            model += conv(channels, channels, kernel_size=3, stride=1, padding=1)
            self.blocks.append(nn.Sequential(*model))

        model = []
        channels_in = channels
        channels *= 2
        model += conv(channels_in, channels, kernel_size=3, stride=2, padding=1)
        model += conv(channels, channels, kernel_size=3, stride=1, padding=1)

        model += [nn.Conv2d(channels, output_nc, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        self.blocks.append(nn.Sequential(*model))


    def forward(self, input):
        outputs = []
        for module in self.blocks:
            input = module(input)
            outputs.append(input)
        return outputs



class SkipNipsDecoder(nn.Module):
    def __init__(self, input_nc, output_nc, n_blocks=4, norm_layer=nn.BatchNorm2d):
        super(SkipNipsDecoder, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d


        def conv(channels_in, channels_out, kernel_size=3, stride=1, padding=1):
            return [nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=use_bias),
                    norm_layer(channels_out),
                    nn.ReLU(True)]

        upsampling = 'bilinear'

        self.n_blocks = n_blocks

        model = []
        channels = 32 * (2**(self.n_blocks-1))
        model += conv(input_nc[0], channels, kernel_size=3, stride=1, padding=1)
        model += conv(channels, channels, kernel_size=3, stride=1, padding=1)
        model += [Interpolate(scale=2, mode=upsampling)]
        self.blocks = nn.ModuleList([nn.Sequential(*model)])

        for i in range(self.n_blocks - 2):
            model = []
            channels_in = channels
            channels /= 2
            model += [nn.Conv2d(channels_in + input_nc[i + 1], channels_in, kernel_size=1, stride=1, padding=0, bias=False)]
            model += conv(channels_in, channels, kernel_size=3, stride=1, padding=1)
            model += conv(channels, channels, kernel_size=3, stride=1, padding=1)
            model += [Interpolate(scale=2, mode=upsampling)]
            self.blocks.append(nn.Sequential(*model))

        model = []
        channels_in = channels
        channels /= 2
        model += [nn.Conv2d(channels_in + input_nc[-1], channels_in, kernel_size=1, stride=1, padding=0, bias=False)]
        model += conv(channels_in, channels, kernel_size=3, stride=1, padding=1)
        model += [nn.Conv2d(channels, output_nc, kernel_size=3, stride=1, padding=1, bias=True)]
        self.blocks.append(nn.Sequential(*model))


    def forward(self, inputs):
        output = self.blocks[0](inputs[0])
        for i in range(1, self.n_blocks):
            output = self.blocks[i](
                torch.cat([output, inputs[i]], dim=1))
        return output


class NipsDecoder(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        super(NipsDecoder, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []

        def conv(channels_in, channels_out, kernel_size=3, stride=1, padding=1):
            return [nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=use_bias),
                    norm_layer(channels_out),
                    nn.ReLU(True)]

        upsampling = 'bilinear'

        channels = 256
        model += conv(input_nc, channels, kernel_size=3, stride=1, padding=1)
        model += conv(channels, channels, kernel_size=3, stride=1, padding=1)

        model += [Interpolate(scale=2, mode=upsampling)]
        channels_in = channels
        channels /= 2
        model += conv(channels_in, channels, kernel_size=3, stride=1, padding=1)
        model += conv(channels, channels, kernel_size=3, stride=1, padding=1)

        model += [Interpolate(scale=2, mode=upsampling)]
        channels_in = channels
        channels /= 2
        model += conv(channels_in, channels, kernel_size=3, stride=1, padding=1)
        model += conv(channels, channels, kernel_size=3, stride=1, padding=1)

        model += [Interpolate(scale=2, mode=upsampling)]
        channels_in = channels
        channels /= 2
        model += conv(channels_in, channels, kernel_size=3, stride=1, padding=1)
        model += [nn.Conv2d(channels, output_nc, kernel_size=3, stride=1, padding=1, bias=True)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)




# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)



class MultiDiscriminator(nn.Module):
    def __init__(self, net):
        super(MultiDiscriminator, self).__init__()
        self.scales = [1, 1./2, 1./4]
        for i in range(len(self.scales)):
            self.add_module('net_%d' % i, net())

    def forward(self, input):
        outputs = []
        for i, scale in enumerate(self.scales):
            input_ = torch.nn.functional.interpolate(
                input, scale_factor=scale, mode='bilinear')
            output = getattr(self, 'net_%d' % i)(input_)
            outputs.append(output)
        return outputs
