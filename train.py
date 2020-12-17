import time
import collections
import numpy as np
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util.util import Timer


if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    iterations = opt.iters_count

    avg_time = collections.deque(maxlen=100)

    while True:
        iter_start_time = time.time()
        iter_data_time = time.time()
        for i, data in enumerate(dataset):
            iterations += 1
            visualizer.reset()

            model.set_input(data)
            t_data = time.time() - iter_data_time

            optim_time = time.time()
            model.optimize_parameters()
            t_optim = time.time() - optim_time

            if iterations % opt.display_freq == 0:
                save_result = iterations % opt.update_html_freq == 0
                visualizer.display_current_results(
                    model.get_current_visuals(), iterations, save_result)

            t = (time.time() - iter_start_time)
            iter_start_time = time.time()
            avg_time.append(t)
            if iterations % opt.print_freq == 0:
                losses = model.get_current_losses()
                samples_frq = float(opt.batch_size) / t
                samples_frq_avg = float(opt.batch_size) / np.mean(avg_time)
                prefix = '%.1f samples/sec %.1f samples/sec (avg) %.2f optim ' % (
                    samples_frq, samples_frq_avg, t_optim)
                visualizer.print_current_losses(
                    iterations, losses, t, t_data, prefix=prefix)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(iterations, opt, losses)

            if iterations % opt.save_latest_freq == 0:
                print('saving the latest model (iterations %d)' % (iterations))
                model.save_networks('latest')

            if iterations % opt.save_iters_freq == 0:
                print('saving the model at iters %d' % (iterations))
                model.save_networks('latest')
                model.save_networks(iterations)

            iter_data_time = time.time()

    # FIXME: should be called at the end of an epoch
    # model.update_learning_rate()
