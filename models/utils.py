import math
from collections import Iterable

import matplotlib.cm as mpl_color_map
import numpy as np
import PIL.Image
import PIL.ImageDraw
import torch
import torch.nn.functional as F
from torch.autograd import Function

import pygame


def extract_points(input):
    """
    input is N x K x H x W "heat map" tensor, N is batch size
    """
    def get_coord(input, other_axis, axis_size):
        # get "x-y" coordinates:
        marg = torch.mean(input, dim=other_axis)  # B,W,NMAP
        prob = F.softmax(marg, dim=2)  # B,W,NMAP
        grid = torch.linspace(-1.0, 1.0, axis_size, dtype=torch.float32, device=input.device)  # W
        grid = grid[None, None]
        point = torch.sum(prob * grid, dim=2)
        return point, prob

    x, _ = get_coord(input, 2, input.shape[3])  # B,NMAP
    y, _ = get_coord(input, 3, input.shape[2])  # B,NMAP
    points = torch.stack([x, y], dim=2)

    return points


def render_points(points, width, height, inv_std=50):
    device = points.device
    mu_x, mu_y = points[:, :, 0:1], points[:, :, 1:2]

    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32, device=device)
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32, device=device)

    mu_y, mu_x = mu_y[..., None], mu_x[..., None]

    y = torch.reshape(y, [1, 1, height, 1])
    x = torch.reshape(x, [1, 1, 1, width])

    g_y = (y - mu_y) ** 2
    g_x = (x - mu_x) ** 2
    dist = (g_y + g_x) * inv_std**2

    g_yx = torch.exp(-dist)
    return g_yx


def get_perpendicular_unit_vector(u):
    negative = torch.Tensor([1, -1]).to(u.device)
    # add dimensions to match u
    negative = torch.reshape(negative, [1] * (u.dim() - 1) + [2])
    u = u[..., [1, 0]] * negative
    u = u / torch.norm(u, dim=-1, keepdim=True)
    return u


def render_line_segment(a, b, size, distance='gauss', sigma=0.2,
                        normalize=False, widths=None):
    """
    a, b points defining the line segment
    widths: B x N
    outputs B x N x H x W
    """
    def sumprod(x, y, keepdim=True):
        return torch.sum(x * y, dim=-1, keepdim=keepdim)

    grid = torch.linspace(-1.0, 1.0, size, dtype=torch.float32, device=a.device)

    # FIXME: api different from numpy
    yv, xv = torch.meshgrid([grid, grid])
    # 1 x H x W x 2
    m = torch.cat([xv[..., None], yv[..., None]], dim=-1)[None, None]

    # B x N x 1 x 1 x 2
    a, b = a[:, :, None, None, :], b[:, :, None, None, :]
    t_min = sumprod(m - a, b - a) / \
        torch.max(sumprod(b - a, b - a), torch.tensor(1e-6, device=a.device))
    t_line = torch.clamp(t_min, 0.0, 1.0)

    # closest points on the line to every image pixel
    s = a + t_line * (b - a)

    # for rectangle
    if widths is not None:
        # get perpendicular unit vector
        u = b - a
        u = u[..., [1, 0]] * torch.Tensor([[[[[1, -1]]]]]).to(u.device)
        u = u / torch.norm(u, dim=-1, keepdim=True)

        t_min = sumprod(m - s, u) / \
            torch.max(sumprod(u, u), torch.tensor(1e-6, device=a.device))

        w = widths[..., None, None, None] / 2.0
        t_line = clamp(t_min, -w, w)

        s = s + t_line * u

    d = sumprod(s - m, s - m, keepdim=False)

    # normalize distancin
    if distance == 'gauss':
        d = torch.sqrt(d + 1e-6)
        d_norm = torch.exp(-d / (sigma ** 2))
    elif distance == 'norm':
        d = torch.sqrt(d + 1e-6)
        d_max = torch.sqrt(8)
        d_norm = (d_max - d) / d_max
    else:
        raise ValueError()

    if normalize:
        d_norm = d_norm / torch.sum(d_norm, (2, 3), keepdim=True)

    return d_norm


def get_line_points(points, connections):
    # gather points for lines
    a_points = torch.zeros(
        [points.shape[0], len(connections), points.shape[2]],
        dtype=points.dtype, device=points.device)
    b_points = torch.zeros_like(a_points)
    for i, (a, b) in enumerate(connections):
        a_points[:, i] = _mean_point(points, a)
        b_points[:, i] = _mean_point(points, b)
    return a_points, b_points


def get_polygons_points(points, polygons):
    polygon_points = []
    for polygon in polygons:
        polygon_points += [get_polygon_points(points, polygon)]
    return polygon_points


def get_polygon_points(points, polygon):
    point_ids = [x for x, _ in polygon]
    polygon_points = points[:, point_ids]
    return polygon_points


def render_skeleton(points, connections, width, height, colored=False,
                    auxilary_links=None, colors=None, reduce=None,
                    sigma=0.2, normalize=False, widths=None):
    """
    colors: B x N x C
    returns: B x N x C x H x W or B x C x H x W if using reduce
    """
    assert width == height

    batch_size = points.shape[0]
    if auxilary_links is not None:
        points, connections = add_auxiliary_links(
            points, connections, auxilary_links)

    # create colors if required
    if colors is None:
        if colored:
            colors = torch.linspace(0.2, 1.0, len(connections), dtype=torch.float32, device=points.device)
            colors = colors[None, :, None].repeat([batch_size, 1, 1])
        else:
            colors = torch.ones([batch_size, len(connections), 1], dtype=torch.float32, device=points.device)

    # parse auxiliary links contained in connectios
    points, connections = parse_auxiliary_links(points, connections)

    # gather points for lines
    a, b = zip(*connections)
    a, b = list(a), list(b)
    a_points = points[:, a]
    b_points = points[:, b]
    
    renderings = render_line_segment(a_points, b_points, width, sigma=sigma,
                                     normalize=normalize, widths=widths)
    # add an axis for colors, renderings has B x N x 1 x H x W
    renderings = renderings[:, :, None]
    renderings = renderings * colors[..., None, None]

    # renderings has B x N x C x H x W
    return renderings


def add_auxiliary_links(points, connections, auxilary_links):
    def mean_point(points, indices):
        if isinstance(indices, Iterable):
            return torch.mean(points[:, indices], dim=1)
        else:
            return points[:, indices]

    # add auxilary points and links
    connections = connections[:]
    n_aux_points = 2 * len(auxilary_links)
    index_offset = points.shape[1]
    zeros = torch.zeros([points.shape[0], n_aux_points, 2],
                        dtype=points.dtype, device=points.device)
    points = torch.cat([points, zeros], dim=1)
    for (a, b), i in zip(auxilary_links, range(index_offset, index_offset + n_aux_points, 2)):
        points[:, i] = mean_point(points, a)
        points[:, i + 1] = mean_point(points, b)
        connections.append([i, i + 1])

    return points, connections


def parse_auxiliary_links(points, connections):
    # add auxilary points and links
    new_connections = []
    n_points = 2 * len(connections)
    new_points = torch.zeros([points.shape[0], n_points, 2],
                        dtype=points.dtype, device=points.device)
    for (a, b), i in zip(connections, range(0, n_points, 2)):
        new_points[:, i] = _mean_point(points, a)
        new_points[:, i + 1] = _mean_point(points, b)
        new_connections.append([i, i + 1])

    return new_points, new_connections


def _mean_point(points, indices):
    if isinstance(indices, Iterable):
        return torch.mean(points[:, indices], dim=1)
    else:
        return points[:, indices]


def normalize_im(im):
    return 2.0 * im - 1.0


def l2_distance(x, y):
    """
    x: B x N x D
    y: B x N x D
    """
    return torch.sqrt(torch.sum((x - y) ** 2, dim=2))


def mean_l2_distance(x, y):
    """
    x: B x N x D
    y: B x N x D
    """
    return torch.mean(l2_distance(x, y), dim=1)


def mean_l2_distance_norm(predict, target, norm_points):
    """
    x: B x N x D
    y: B x N x D
    """
    dists = l2_distance(predict, target)
    dists_points = l2_distance(
        target[:, norm_points[0]][:, None],
        target[:, norm_points[1]][:, None])
    norm_dists = dists / dists_points
    return torch.mean(norm_dists, dim = 1)


def swap_points(points, correspondences):
    """
    points: B x N x D
    """
    permutation = list(range((points.shape[1])))
    for a, b in correspondences:
        permutation[a] = b
        permutation[b] = a
    new_points = points[:, permutation, :]
    return new_points


def normalize_image_tensor(tensor):
    minimum, _ = torch.min(tensor, dim=2, keepdim=True)
    minimum, _ = torch.min(minimum, dim=3, keepdim=True)
    tensor -= minimum
    maximum, _ = torch.max(tensor, dim=2, keepdim=True)
    maximum, _ = torch.max(maximum, dim=3, keepdim=True)
    return tensor / maximum


def clamp(tensor, minimum, maximum):
    tensor = torch.max(tensor, minimum)
    tensor = torch.min(tensor, maximum)
    return tensor


def lstsq(input, A):
    solution = []
    QR = []
    for i, a in zip(input, A):
        s, q = torch.lstsq(i, a)
        solution += [s]
        QR += [q]
    solution = torch.stack(solution)
    QR = torch.stack(QR)
    return solution, QR


def rollout(tensor):
    """
    tensor: B x C x .... -> B * C x ...
    """
    shape = tensor.shape
    new_shape = [shape[0] * shape[1]]
    if len(shape) > 2:
        new_shape += shape[2:]
    return torch.reshape(tensor, new_shape)


def unrollout(tensor, n_channels):
    """
    tensor: B * C x ... -> B x C x ...
    """
    shape = tensor.shape
    new_shape = [shape[0] // n_channels, n_channels]
    if len(shape) > 1:
        new_shape += shape[1:]
    return torch.reshape(tensor, new_shape)


def apply_colormap_on_tensor(tensor, colormap_name='jet'):
    """
    """
    # Get colormap
    assert tensor.shape[1] == 1

    color_map = mpl_color_map.get_cmap(colormap_name)
    
    tensor = normalize_tensor_image(tensor, dim=(2, 3))
    tensor_np = tensor.detach().cpu().numpy()
    heatmap = color_map(tensor_np[:, 0])
    # remove alpha
    heatmap = heatmap[:, :, :, :3]
    heatmap = torch.from_numpy(heatmap).to(tensor.device).type(tensor.dtype)
    heatmap = heatmap.permute(0, 3, 1, 2)

    return heatmap


def normalize_tensor_image(tensor, dim):
    minv = multi_min(tensor, dim, keepdim=True)
    maxv = multi_max(tensor, dim, keepdim=True)
    return (tensor - minv) / (maxv - minv)



def multi_min(input, dim, keepdim=False):
    return _multi_minmax(input, dim, torch.min, keepdim=keepdim)


def multi_max(input, dim, keepdim=False):
    return _multi_minmax(input, dim, torch.max, keepdim=keepdim)


def _multi_minmax(input, dim, operator, keepdim=False):
    dim = sorted(dim)
    reduced = input
    for axis in reversed(dim):
        reduced, _ = operator(reduced, axis, keepdim=keepdim)
    return reduced


# https://gist.github.com/bobchennan/a865b153c6835a3a6a5c628213766150
class gels(Function):
    """ Efficient implementation of gels from
        Nanxin Chen
        bobchennan@gmail.com
    """
    @staticmethod
    def forward(ctx, A, b):
        # A: (..., M, N)
        # b: (..., M, K)
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/linalg_ops.py#L267
        u = torch.cholesky(torch.matmul(A.transpose(-1, -2), A), upper=True)
        ret = torch.cholesky_solve(torch.matmul(A.transpose(-1, -2), b), u, upper=True)
        ctx.save_for_backward(u, ret, A, b)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/linalg_grad.py#L223
        chol, x, a, b = ctx.saved_tensors
        z = torch.cholesky_solve(grad_output, chol, upper=True)
        xzt = torch.matmul(x, z.transpose(-1,-2))
        zx_sym = xzt + xzt.transpose(-1, -2)
        grad_A = - torch.matmul(a, zx_sym) + torch.matmul(b, z.transpose(-1, -2))
        grad_b = torch.matmul(a, z)
        return grad_A, grad_b
