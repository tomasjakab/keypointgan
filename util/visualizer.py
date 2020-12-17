import numpy as np
import os
import sys
import ntpath
import time
import torch
import re
from . import util
from . import html
from skimage.transform import resize
from PIL import Image
from models.utils import normalize_image_tensor
from collections import OrderedDict
import skvideo.io


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


# save image to the disk
def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256,
                basename=True):
    # preprocess multichannel maps
    for label, image in visuals.items():
        if image.shape[1] not in [1, 3] and isinstance(image, torch.Tensor):
            max_frames = 20
            for i in range(min(max_frames, image.shape[1])):
                visuals[label + str(i)] = image[:, i][:, None]
            visuals[label], _ = torch.max(image, dim=1, keepdim=True)

    image_dir = webpage.get_image_dir()
    if basename:
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]
    else:
        name = image_path[0]

    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        if label == 'fake_B':
            tensor = np.transpose(im_data[0].cpu().numpy(), (1, 2, 0))
            tensor_name = '%s-%s.npy' % (name, label)
            np.save(os.path.join(image_dir, tensor_name), tensor)

        im = util.tensor2im(im_data)
        image_name = '%s-%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = resize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im = resize(im, (int(h / aspect_ratio), w), interp='bicubic')
        util.save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_media(ims, txts, links, 'image', width=width, title=name)


def save_videos(webpage, visuals_log, image_path, width=256, basename=True):
    image_dir = webpage.get_image_dir()
    if basename:
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]
    else:
        name = image_path[0]

    ims, txts, links = [], [], []

    for label in visuals_log.keys():
        video_name = '%s-%s.mp4' % (name, label)
        save_path = os.path.join(image_dir, video_name)
        visuals_log.save_as_video(label, save_path)
        ims.append(video_name)
        txts.append(label)
        links.append(video_name)

    webpage.add_media(ims, txts, links, 'video', width=width, title=name)


class Visualizer():
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env, raise_exceptions=True)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def throw_visdom_connection_error(self):
        print('\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n')
        exit(1)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, iteration, save_result):
        visuals = preprocess_multi_channel(visuals)

        if self.display_id > 0:  # show images in the browser
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.throw_visdom_connection_error()

            else:
                idx = 1
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'latest_%s.png' % (label))
                util.save_image(image_numpy, img_path)
                if iteration % self.opt.save_iters_freq == 0:
                    img_path = os.path.join(
                        self.img_dir, 'iteration%.7d_%s.png' % (iteration, label))
                    util.save_image(image_numpy, img_path)

            # find saved images
            img = os.listdir(self.img_dir)
            reg = re.compile('(iteration0*[0-9]+)*')
            saved_iterations = set()
            for img in os.listdir(self.img_dir):
                match = reg.match(img).group(1)
                if match:
                    saved_iterations.add(match)
            saved_iterations = list(saved_iterations)
            saved_iterations = sorted(saved_iterations, reverse=True)
            prefixes = ['latest'] + saved_iterations
            # update website
            webpage = html.HTML(self.web_dir, '%s' % self.name, reflesh=0)
            for prefix in prefixes:
                ims, txts, links = [], [], []
                for label, _ in visuals.items():
                    # convert images to png
                    full_img_path_png = os.path.join(
                        webpage.get_image_dir(), '%s_%s.png' % (prefix, label))
                    full_img_path_jpg = os.path.join(
                        webpage.get_image_dir(), '%s_%s.jpg' % (prefix, label))
                    if os.path.isfile(full_img_path_jpg) and not os.path.isfile(full_img_path_png):
                        image = Image.open(full_img_path_jpg)
                        image.save(full_img_path_png)
                    img_path = '%s_%s.png' % (prefix, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_media(ims, txts, links, 'image', width=self.win_size, title='%s' % prefix)
            webpage.save()

    # losses: dictionary of error labels and values
    def plot_current_losses(self, iteration, opt, losses):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(iteration)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.throw_visdom_connection_error()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, iteration, losses, t, t_data, prefix='', txt=None):
        message = prefix + '(iters: %d, time: %.3f, data: %.3f) ' % (
            iteration, t, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        if txt is not None:
            message += ' ' + txt
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)


def preprocess_multi_channel(visuals):
    # preprocess multichannel maps
    new_visuals = {}
    for label, image in visuals.items():
        if image.shape[1] not in [1, 3] or len(image.shape) == 5:
            max_frames = 20
            for i in range(min(max_frames, image.shape[1])):
                if len(image.shape) == 5:
                    frame = image[:, i]
                else:
                    frame = normalize_image_tensor(image[:, i][:, None])
                new_visuals[label + '_' + str(i)] = frame
            # visuals[label], _ = torch.max(image, dim=1, keepdim=True)
        else:
            new_visuals[label] = image
    return new_visuals


class VisualsLog(object):

    def __init__(self):
        self.log = OrderedDict()
    
    
    def append(self, visuals):
        visuals = preprocess_multi_channel(visuals)
        for name, visual in visuals.items():
            visual = visual.to('cpu')
            if name not in self.log:
                self.log[name] = []
            self.log[name] += [visual]
    

    def save_as_video(self, visual_name, path):
        images = self.log[visual_name]
        images = [util.tensor2im(x) for x in images]
        skvideo.io.vwrite(path, images, outputdict={'-crf': '1', '-pix_fmt': 'yuv420p'})

    def keys(self):
        return self.log.keys()
