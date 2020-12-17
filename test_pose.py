import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from util import util
import torch
from models import utils as mutils
from data import human36m_skeleton
import math
import itertools
import torch
from collections import defaultdict
import numpy as np
import time
import re


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_test = 1000000
    num_save = opt.num_test_save

    # load data
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    print('Created dataset with %d samples' % len(dataset))

    # setup model
    model = create_model(opt)
    model.setup(opt)

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.iteration))
    webpage = html.HTML(web_dir, '%s, Phase = %s, Iteration = %s' % (
        opt.name, opt.phase, opt.iteration))

    if opt.eval:
        model.eval()

    ############################################################################
    def format_results_per_activity(results):
        """
        results dict {activity: result}
        """
        s = ''
        for activity, result in sorted(results.items()):
            s += '%s: %.4f ' % (activity, result)
        return s


    def format_results_per_activity2(results):
        """
        results dict {activity: result}
        """
        order_full = ['waiting', 'posing', 'greeting', 'directions', 'discussion', 'walking',
                'eating', 'phone_call', 'purchases', 'sitting', 'sitting_down', 'smoking', 
                'taking_photo', 'walking_dog', 'walking_together']
        order_yutig = ['Waiting', 'Posing', 'Greeting', 'Directions', 'Discussion', 'Walking']
        
        if set(order_full) == set(results.keys()):
            order = order_full
        elif set(order_yutig) == set(results.keys()):
            order = order_yutig
        else:
            raise ValueError()

        numbers = ['%.4f' % results[k] for k in order]
        return '\t'.join(order), '\t'.join(numbers)


    def mean_per_activity(distances, paths, path_fn):
        activities = [path_fn(p) for p in paths]
        return mean_distance_per_activity(distances, activities)


    def human36m_path_to_activity(path):
        return path.split(os.path.sep)[-5]


    def y_human36m_path_to_activity(path):
        return re.split('\s|\.', path.split(os.path.sep)[-2])[0]


    def mean_distance_per_activity(distances, activities):
        d = defaultdict(list)
        for distance, activity in zip(distances, activities):
            d[activity].append(distance)
        means = {}
        for activity, values in d.items():
            means[activity] = np.mean(values)
        return means


    def compute_mean_distance(input, target, correspondeces=None, 
                              target_correspondeces=None, used_points=None,
                              offline_prediction=None):
        if target_correspondeces is not None:
            target_swapped = mutils.swap_points(target, target_correspondeces)
        else:
            target_swapped = target.clone()
        if correspondeces is not None:
            input_swapped = mutils.swap_points(input, correspondeces)
            if offline_prediction is not None:
                offline_prediction_swapped = mutils.swap_points(offline_prediction, correspondeces)
        else:
            input_swapped = input.clone()
            if offline_prediction is not None:
                offline_prediction_swapped = offline_prediction.clone()

        if used_points is not None:
            input = input[:, used_points]
            input_swapped = input_swapped[:, used_points]
            if offline_prediction is not None:
                offline_prediction = offline_prediction[:, used_points]
            target = target[:, used_points]
            target_swapped = target_swapped[:, used_points]

        # offline
        if offline_prediction is not None:
            distance = mutils.mean_l2_distance(offline_prediction, input)
            swapped_distance = mutils.mean_l2_distance(offline_prediction, input_swapped)
            min_idx = distance > swapped_distance
            for i in range(len(min_idx)):
                if min_idx[i]:
                    input[i] = input_swapped[i]

        distance = mutils.mean_l2_distance(target, input)
        swapped_distance = mutils.mean_l2_distance(target_swapped, input)
        correct_flip = distance < swapped_distance
        min_distance = torch.min(distance, swapped_distance)

        return distance, min_distance, correct_flip

    def normalize_points(points):
        return (points + 1) / 2.0

    def points_to_original(points, height, width, height_ratio, width_ratio):
        """
        points: B x N x 2
        """
        points *= torch.tensor([[[height, width]]], dtype=torch.float32, device=points.device)
        points /= torch.stack([height_ratio, width_ratio], dim=-1, )[:, None].to(points.device)
        return points

    if opt.used_points == 'simple_links':
        used_points = set()
        for a, b in human36m_skeleton.simple_link_indices:
            used_points.add(a)
            used_points.add(b)
        used_points = list(used_points)
        used_points = sorted(used_points)
    elif opt.used_points == 'original':
        used_points = sorted(list(human36m_skeleton.official_eval_indices.values()))
    elif opt.used_points in ['all']:
        used_points = None
    else:
        raise ValueError()

    ############################################################################

    distances = []
    distances_min = []
    correct_flips = []
    paths = []

    if opt.paired_skeleton_type in ['human36m', 'human36m_simple2']:
        target_correspondeces = human36m_skeleton.get_lr_correspondences()
    else:
        target_correspondeces = None

    if opt.skeleton_type in ['human36m', 'human36m_simple2']:
        correspondeces = human36m_skeleton.get_lr_correspondences()
    else:
        correspondeces = None

    n_batches = int(math.ceil(float(len(dataset)) / opt.batch_size))

    save_frq = int(math.ceil(float(min(n_batches, opt.num_test)) / num_save))

    avg_time = []

    path_fn = human36m_path_to_activity
    if opt.dataset_mode == 'simplehuman36m':
        path_fn = y_human36m_path_to_activity

    iter_start_time = time.time()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break

        model.set_input(data)
        model.test()

        img_path = model.get_image_paths()

        prediction = model.regressed_points
        if hasattr(model, 'offline_regressed_points'):
            offline_prediction = model.offline_regressed_points
        else:
            offline_prediction = None
        target = model.paired_B_points

        prediction = normalize_points(prediction)
        if offline_prediction is not None:
            offline_prediction = normalize_points(offline_prediction)
        target = normalize_points(target)

        if opt.error_form == 'original':
            height_ratio = model.input['height_ratio']
            width_ratio = model.input['width_ratio']
            original_landmarks = model.input['landmarks']
            prediction = points_to_original(
                prediction, opt.fineSize, opt.fineSize, height_ratio, width_ratio)
            if offline_prediction is not None:
                offline_prediction = points_to_original(
                    offline_prediction, opt.fineSize, opt.fineSize, height_ratio, width_ratio)
            target = points_to_original(
                target, opt.fineSize, opt.fineSize, height_ratio, width_ratio)
        elif opt.error_form == 'image_size':
            pass
        else:
            raise ValueError()

        # compute distance error
        dist, dist_min, correct_flip = compute_mean_distance(
            prediction, target, correspondeces=correspondeces,
            target_correspondeces=target_correspondeces,
            used_points=used_points, offline_prediction=offline_prediction)

        # log results
        distances.extend(dist.cpu().numpy())
        distances_min.extend(dist_min.cpu().numpy())
        correct_flips.extend(correct_flip.cpu().numpy())
        paths.extend(data['A_paths'])

        t = (time.time() - iter_start_time)
        iter_start_time = time.time()
        avg_time.append(t)
        if i % opt.print_freq == 0:
            samples_frq = float(opt.batch_size) / t
            samples_frq_avg = float(opt.batch_size) / np.mean(avg_time)
            time_str = '%.1f samples/sec %.1f samples/sec (avg)' % (
                samples_frq, samples_frq_avg)

            print('processing (%d/%d)-th batch %s' % (i, n_batches, time_str))
            print(np.random.choice(img_path, 1))

            mean_distances = mean_per_activity(distances, paths, path_fn)
            mean_distance = np.mean(mean_distances.values())
            mean_min_distances = mean_per_activity(distances_min, paths, path_fn)
            mean_min_distance = np.mean(mean_min_distances.values())
            mean_correct_flips = mean_per_activity(correct_flips, paths, path_fn)
            mean_correct_flip = np.mean(mean_correct_flips.values())

            results_str = 'mean distance %.4f\n' % mean_distance
            results_str += 'mean min distance %.4f\n' % mean_min_distance
            results_str += 'mean correct flips %.4f\n' % mean_correct_flip
            results_str += '%s\n' % format_results_per_activity(mean_distances)
            results_str += '%s\n' % format_results_per_activity(mean_min_distances)
            results_str += '%s\n' % format_results_per_activity(mean_correct_flips)
            print(results_str)

        if i % save_frq == 0:
            visuals = model.get_current_visuals()
            webpage.add_text(data['A_paths'][0])
            if opt.dataset_mode == 'simplehuman36m':
                data['image_name'] = ['-'.join(x.split(os.path.sep)[-2:]) for x in img_path]
            if 'image_name' in data:
                img_names = [x.replace(os.path.sep, '-') for x in data['image_name']]
                take_basename = False
            else:
                img_names = img_path
                take_basename = True
            save_images(
                webpage, visuals, img_names, aspect_ratio=opt.aspect_ratio,
                width=opt.display_winsize, basename=take_basename)
            webpage.save()

    mean_distances = mean_per_activity(distances, paths, path_fn)
    mean_distance = np.mean(mean_distances.values())
    mean_min_distances = mean_per_activity(distances_min, paths, path_fn)
    mean_min_distance = np.mean(mean_min_distances.values())
    mean_correct_flips = mean_per_activity(correct_flips, paths, path_fn)
    mean_correct_flip = np.mean(mean_correct_flips.values())

    results_str = 'mean distance %.4f\n' % mean_distance
    results_str += 'mean min distance %.4f\n' % mean_min_distance
    results_str += 'mean correct flips %.4f\n' % mean_correct_flip
    results_str += '%s\n' % format_results_per_activity(mean_distances)
    results_str += '%s\n' % format_results_per_activity(mean_min_distances)
    results_str += '%s\n' % format_results_per_activity(mean_min_distances)
    results_str += '%s\n' % format_results_per_activity(mean_correct_flips)

    print(results_str)
    webpage.add_text(results_str)

    results_str = format_results_per_activity2(mean_distances)
    print(results_str[0])
    webpage.add_text(results_str[0])
    print(results_str[1])
    webpage.add_text(results_str[1])

    results_str = format_results_per_activity2(mean_min_distances)
    print(results_str[0])
    webpage.add_text(results_str[0])
    print(results_str[1])
    webpage.add_text(results_str[1])

    # save the website
    webpage.save()
