import numpy as np
from scipy.stats import truncnorm


def render_line_segment(s1, s2, size, distance='gauss', discrete=False):
    def sumprod(x, y):
        return np.sum(x * y, axis=-1, keepdims=True)

    x = np.linspace(-1.0, 1.0, size).astype('float32')
    y = np.linspace(-1.0, 1.0, size).astype('float32')

    xv, yv = np.meshgrid(x, y)
    m = np.concatenate([xv[..., None], yv[..., None]], axis=-1)

    s1, s2 = s1[None, None], s2[None, None]
    t_min = sumprod(m - s1, s2 - s1) / np.maximum(sumprod(s2 - s1, s2 - s1), 1e-6)
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


def normalize_landmarks(landmarks):
    """
    Centre and stretch landmarks, preserve aspect ratio
    Landmarks are [[y_0, x_0], [y_1, x_1], ...]
    """
    ymin, ymax = min(landmarks[:, 0]), max(landmarks[:, 0])
    xmin, xmax = min(landmarks[:, 1]), max(landmarks[:, 1])

    # put in the corner
    landmarks -= np.min(landmarks, axis=0, keepdims=True)
    # normalize between -1, 1
    height, width = np.max(landmarks, axis=0)
    landmarks = 2.0 * (landmarks / max(height, width)) - 1.0
    # centre
    landmarks += (1.0 - np.max(landmarks, axis=0, keepdims=True)) / 2.0
    return landmarks


def rotate_points(points, angle):
    rot = np.deg2rad(angle)
    af = [[ np.cos(rot), np.sin(rot), 0],
          [-np.sin(rot), np.cos(rot), 0]]
    af = np.array(af, dtype=np.float32)
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    points = np.concatenate([points, ones], axis=1)
    points = np.matmul(points, af.T)
    return points


def jitter_landmarks(landmarks, zoom=[0.5, 1.0], aspect_ratio=[1.0, 1.0],
                     shift=True, rotate=[0.0, 0.0]):
    """
    expects points normalized in [-1, 1]
    """
    # rotate and refit to the canvas
    if rotate != [0.0, 0.0]:
        angle = np.random.uniform(rotate[0], rotate[1])
        landmarks = rotate_points(landmarks, angle)
        landmarks = normalize_landmarks(landmarks)

    # zoom
    if zoom != [1.0, 1.0]:
        # generate random number between 0.0 and 1.0 (1.0 has higher probability)
        rand = 1 + truncnorm.rvs(-1.0, 0.0)
        zoom_ratio = zoom[0] + (zoom[1] - zoom[0]) * rand
        landmarks *= zoom_ratio

    # aspect ratio
    if aspect_ratio != [1.0, 1.0]:
        landmarks[:, 0] *= np.random.uniform(aspect_ratio[0], aspect_ratio[1])

    # shift but keep all in the canvas
    if shift:
        shift_y = [-1 - min(landmarks[:, 0]),  1 - max(landmarks[:, 0])]
        shift_x = [-1 - min(landmarks[:, 1]),  1 - max(landmarks[:, 1])]
        landmarks[:, 0] += np.random.uniform(shift_y[0], shift_y[1])
        landmarks[:, 1] += np.random.uniform(shift_x[0], shift_x[1])

    return landmarks


def pad_landmarks(landmarks, ratio):
    """
    landmarks normalize in [-1, 1]
    ratio in [0, 1]
    """
    return landmarks * (1.0 / (1 + 2 * ratio))


def landmarks_to_image_space(landmarks, height, width):
    landmarks = ((landmarks + 1.0) / 2) * min(height, width)
    return landmarks
