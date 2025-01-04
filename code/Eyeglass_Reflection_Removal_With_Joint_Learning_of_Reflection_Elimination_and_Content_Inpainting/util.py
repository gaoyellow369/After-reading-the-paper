from __future__ import print_function
import os
import sys
import time

import torch
import numpy as np
import yaml
from PIL import Image


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    assert len(image_numpy.shape) in [2, 3] or image_numpy.size == 0
    if len(image_numpy.shape) == 3:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
        image_numpy = image_numpy.astype(imtype)
    else:
        image_numpy = image_numpy * 255.0
        image_numpy = image_numpy.astype(imtype)
    return image_numpy


def tensor2numpy(image_tensor):
    image_numpy = torch.squeeze(image_tensor).cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.float32)
    return image_numpy


# Get model list for resume
def get_model_list(dirname, key, epoch=None):
    if epoch is None:
        return os.path.join(dirname, key + '_latest.pt')
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f and 'latest' not in f]
    if gen_models is None:
        return None

    epoch_index = [int(os.path.basename(model_name).split('_')[-2]) for model_name in gen_models if
                   'latest' not in model_name]
    print('[i] available epoch list: %s' % epoch_index, gen_models)
    i = epoch_index.index(int(epoch))

    return gen_models[i]


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_opt_param(optimizer, key, value):
    for group in optimizer.param_groups:
        group[key] = value


def vis(x):
    if isinstance(x, torch.Tensor):
        Image.fromarray(tensor2im(x)).show()
    elif isinstance(x, np.ndarray):
        Image.fromarray(x.astype(np.uint8)).show()
    else:
        raise NotImplementedError('vis for type [%s] is not implemented', type(x))


"""tensorboard"""
from tensorboardX import SummaryWriter
from datetime import datetime


def get_summary_writer(log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)
    return writer


class AverageMeters(object):
    def __init__(self, dic=None, total_num=None):
        self.dic = dic or {}
        # self.total_num = total_num
        self.total_num = total_num or {}

    def update(self, new_dic):
        for key in new_dic:
            if not key in self.dic:
                self.dic[key] = new_dic[key]
                self.total_num[key] = 1
            else:
                self.dic[key] += new_dic[key]
                self.total_num[key] += 1
        # self.total_num += 1

    def __getitem__(self, key):
        return self.dic[key] / self.total_num[key]

    def __str__(self):
        keys = sorted(self.keys())
        res = ''
        for key in keys:
            res += (key + ': %.4f' % self[key] + ' | ')
        return res

    def keys(self):
        return self.dic.keys()


def write_loss(writer, prefix, avg_meters, iteration):
    for key in avg_meters.keys():
        meter = avg_meters[key]
        writer.add_scalar(
            os.path.join(prefix, key), meter, iteration)


"""progress bar"""
import socket

# 获取控制台大小
# _, term_width = os.popen('stty size', 'r').read().split()
term_width = 80
term_width = int(term_width)
# term_width = 80

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


# 在命令行中显示进度条
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def parse_args(args):
    str_args = args.split(',')
    parsed_args = []
    for str_arg in str_args:
        arg = int(str_arg)
        if arg >= 0:
            parsed_args.append(arg)
    return parsed_args
