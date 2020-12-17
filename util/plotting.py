

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.transforms
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
import shutil

from PIL import Image


def get_marker_style(i, cmap='Dark2'):
  cmap = plt.get_cmap(cmap)
  colors = [cmap(c) for c in np.linspace(0., 1., 8)]
  markers = ['v', 'o', 's', 'd', '^', 'x', '+']
  max_i = len(colors) * len(markers) - 1
  if i > max_i:
    raise ValueError('Exceeded maximum (' + str(max_i) + ') index for styles.')
  c = i % len(colors)
  m = int(i / len(colors))
  return colors[c], markers[m]


def single_marker_style(color, marker):
  return lambda _: (color, marker)


def plot_line(ax, a, b, k, size=1.5, zorder=2, cmap='Dark2',
                  style_fn=None):
  if style_fn is None:
    c, _ = get_marker_style(k, cmap=cmap)
  else:
    c, _ = style_fn(k)
  line = ax.plot([a[0], b[0]], [a[1], b[1]], c=c, zorder=zorder, linewidth=10)
  plt.setp(line, linewidth=5)


def plot_lines(ax, lines, size=1.5, zorder=2, cmap='Dark2', style_fn=None):
  # TODO: avoid for loop if possible
  for k, (a, b) in enumerate(lines):
    plot_line(ax, a, b, k, size=size, zorder=zorder,
              cmap=cmap, style_fn=style_fn)

def plot_landmark(ax, landmark, k, size=1.5, zorder=2, cmap='Dark2',
                  style_fn=None):
  if style_fn is None:
    c, m = get_marker_style(k, cmap=cmap)
  else:
    c, m = style_fn(k)
  ax.scatter(landmark[1], landmark[0], c=c, marker=m,
             s=(size * mpl.rcParams['lines.markersize']) ** 2,
             zorder=zorder)


def plot_landmarks(ax, landmarks, size=1.5, zorder=2, cmap='Dark2', style_fn=None):
  # TODO: avoid for loop if possible
  for k, landmark in enumerate(landmarks):
    plot_landmark(ax, landmark, k, size=size, zorder=zorder,
                  cmap=cmap, style_fn=style_fn)


def show_landmarks(image, landmarks, save_path, landmark_size=1.5,
                   style='uniform', color='limegreen', connections=None):
  def plt_start():
    fig = plt.figure(figsize=(4, 4), dpi=80)
    ax = plt.gca()
    return fig, ax

  def plt_finish(ax, fig, path):
    ax.set_ylim([sz[0], 0])
    ax.set_xlim([0, sz[1]])
    plt.tight_layout()
    plt.autoscale(tight=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_axis_off()
    ax.set_aspect('equal')
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    plt.savefig(path, bbox_inches=matplotlib.transforms.Bbox.from_extents(
        [0, 0, 4, 4]), pad_inches=0)
    plt.close(fig.number)

  sz = image.shape[:2]
  landmarks_scaled = ((landmarks + 1) / 2.0) * sz
  landmarks_scaled = np.clip(landmarks_scaled, 6, sz[0] - 6)

  save_dir = []
  fig, ax = plt_start()
  ax.imshow(image)
  if style == 'uniform':
    ax.scatter(landmarks_scaled[:, 0].T, landmarks_scaled[:, 1].T, c=color,
               s=(landmark_size * mpl.rcParams['lines.markersize']) ** 2)
  elif style == 'skeleton':
    lines = []
    for a, b in connections:
      lines.append((landmarks_scaled[a], landmarks_scaled[b]))
    plot_lines(ax, lines, size=landmark_size)
  else:
    plot_landmarks(ax, landmarks_scaled, size=landmark_size)
  plt_finish(ax, fig, save_path)


def plot_in_image(image, landmarks, landmark_size=1.3, color='limegreen',
                  style='uniform', connections=None):
  tempdir_path = tempfile.mkdtemp()
  tempfile_path = os.path.join(tempdir_path, 'im.png')
  try:
    show_landmarks(image, landmarks, tempfile_path,
                   landmark_size=landmark_size, color=color,
                   connections=connections, style=style)
    plot = np.array(Image.open(tempfile_path))
  finally:
    shutil.rmtree(tempdir_path)
  return plot


