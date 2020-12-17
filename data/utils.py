import numpy as np
import skimage
import torchvision.transforms as transforms



def find_common_box(boxes):
    """
    Finds the union of boxes, represented as [xmin, ymin, xmax, ymax].
    """
    boxes = np.stack(boxes, axis=0)
    box = np.concatenate([np.min(boxes[:, :2], axis=0),
                          np.max(boxes[:, 2:], axis=0)], axis=0)
    return box


def fit_box(box, width, height):
    """
    Ajusts box size to have the same aspect ratio as the target image
    while preserving the centre.
    """
    box = box.astype('float32')
    im_w, im_h = float(width), float(height)
    w, h = box[2] - box[0], box[3] - box[1]

    # r_im - image aspect ratio, r - box aspect ratio
    r_im = im_w / im_h
    r = w / h

    centre = [box[0] + w / 2, box[1] + h / 2]

    if r < r_im:
        h, w = h, r_im * h
    else:
        h, w = (1 / r_im) * w, w

    box = [centre[0] - w / 2, centre[1] - h / 2,
           centre[0] + w / 2, centre[1] + h / 2]

    box = np.array(box, dtype='int32')
    return box


def crop_to_box(image, bbox, pad=True):
    bbox = bbox.astype('int32')
    if pad:
        sz = image.shape[:2]
        pad_top = -min(0, bbox[1])
        pad_left = -min(0, bbox[0])
        pad_bottom = -min(0, sz[0] - bbox[3])
        pad_right = -min(0, sz[1] - bbox[2])
        image = np.pad(
            image, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
            'constant')
        bbox[1], bbox[3] = bbox[1] + pad_top,  bbox[3] + pad_top
        bbox[0], bbox[2] = bbox[0] + pad_left, bbox[2] + pad_left
    image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return image


def get_crop_size(box, pad=True):
    box = box.copy()
    if pad:
        pad_top = -min(0, box[1])
        pad_left = -min(0, box[0])
        box[1], box[3] = box[1] + pad_top,  box[3] + pad_top
        box[0], box[2] = box[0] + pad_left, box[2] + pad_left
    return box[2] - box[0], box[3] - box[1]


def resize_points(points, width, height, target_width, target_height):
    dtype = points.dtype
    ratio = np.array([target_width, target_height], dtype='float32') / \
        np.array([width, height], dtype='float32')
    points = (points.astype('float32') * ratio[None]).astype(dtype)
    return points


def box_from_points(points):
    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    return np.concatenate([min_xy, max_xy], axis=0)


def swap_xy_box(box):
    box[:] = box[[1, 0, 3, 2]]
    return box


def swap_xy_points(points):
    points[:, :] = points[:, [1, 0]]
    return points


def normalize_points(points, width, height):
    return 2.0 * points / np.array([width, height], dtype='float32') - 1.0


def render_gaussian_maps(mu, shape_hw, inv_std, mode='rot'):
    """
    Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
    given the gaussian centers: MU [B, NMAPS, 2] tensor.

    STD: is the fixed standard dev.
    """
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]

    y = np.linspace(-1.0, 1.0, shape_hw[0]).astype('float32')

    x = np.linspace(-1.0, 1.0, shape_hw[1]).astype('float32')

    mu_y, mu_x = mu_y[..., None], mu_x[..., None]

    y = np.reshape(y, [1, 1, shape_hw[0], 1])
    x = np.reshape(x, [1, 1, 1, shape_hw[1]])

    g_y = np.square(y - mu_y)
    g_x = np.square(x - mu_x)
    dist = (g_y + g_x) * inv_std**2

    if mode == 'rot':
        g_yx = np.exp(-dist)
    else:
        g_yx = np.exp(-np.power(dist + 1e-5, 0.25))

    g_yx = np.transpose(g_yx, axes=[0, 2, 3, 1])
    return g_yx


def render_points(points, width, height):
    points = normalize_points(points, width, height)
    maps = render_gaussian_maps(
        swap_xy_points(points)[None], [height, width], 50)
    maps = maps[0]
    maps *= np.max(maps)
    return maps


def render_line_segment(s1, s2, size, distance='gauss', discrete=False):
    def sumprod(x, y):
        return np.sum(x * y, axis=-1, keepdims=True)

    x = np.linspace(-1.0, 1.0, size).astype('float32')
    y = np.linspace(-1.0, 1.0, size).astype('float32')

    xv, yv = np.meshgrid(x, y)
    m = np.concatenate([xv[..., None], yv[..., None]], axis=-1)

    s1, s2 = s1[None, None], s2[None, None]
    t_min = sumprod(m - s1, s2 - s1) / \
        np.maximum(sumprod(s2 - s1, s2 - s1), 1e-6)
    t_line = np.minimum(np.maximum(t_min, 0.0), 1.0)

    s = s1 + t_line * (s2 - s1)
    d = np.sqrt(sumprod(s - m, s - m))

    if discrete:
        distance = 'norm'

    # normalize distance
    if distance == 'gauss':
        d_norm = np.exp(-d / (0.2 ** 2))
    elif distance == 'norm':
        d_max = np.sqrt(8)
        d_norm = (d_max - d) / d_max
    else:
        raise ValueError()

    thick = 0.9925
    if discrete:
        d_norm[d_norm >= thick] = 1.0
        d_norm[d_norm < thick] = 0.0

    return d_norm


def render_skeleton(points, connections, width, height, colored=False):
    assert width == height
    maps = []
    numbers = np.linspace(0.2, 1.0, len(connections))
    discrete = False
    if colored:
        discrete = True
    for (a, b), number in zip(connections, numbers):
        render = render_line_segment(
            points[a], points[b], width, discrete=discrete)
        if colored:
            render *= number
        maps.append(render)
    maps = np.concatenate(maps, axis=-1)
    return maps


def proc_im(image, box, landmarks, target_width, target_height, keep_aspect=True, load_image=True):
    # read image
    if load_image:
        image = skimage.io.imread(image)
        if len(image.shape) == 2:
            image = np.tile(image[..., None], (1, 1, 3))

    # crop to bounding box
    if keep_aspect:
        box = fit_box(box, target_width, target_height)
    if load_image:
        image = crop_to_box(image, box)
    else:
        width, height = get_crop_size(box)
    if landmarks is not None:
        landmarks = landmarks - box[:2][None].astype(landmarks.dtype)
    
    # resize
    if landmarks is not None:
        if load_image:
            height, width = image.shape[:2]
        landmarks = resize_points(
            landmarks, height, width, target_width, target_height)

    if load_image:
        image = skimage.transform.resize(image, [target_height, target_width])
        image = skimage.img_as_float(image).astype(np.float32)

    height_ratio, width_ratio = float(target_height) / height, float(target_width) / width

    return image, landmarks, height_ratio, width_ratio


def get_transform(opt, channels=3, normalize=True):
    mean = 0.5
    std = 0.5
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize([mean] * channels,
                           [std] * channels)]
    return transforms.Compose(transform_list)
