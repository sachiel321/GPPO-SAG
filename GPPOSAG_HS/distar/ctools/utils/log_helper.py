'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. log helper, used to help to save logger on terminal, tensorboard or save file.
    2. CountVar, to help counting number.
'''
import json
import logging
import numbers
import os
import sys
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
import yaml
from tabulate import tabulate
from tensorboardX import SummaryWriter


def build_logger(cfg, name=None, rank=0):
    r'''
    Overview:
        use config to build checkpoint helper. Only rank == 0 can build.
    Arguments:
        - name (:obj:`str`): the logger file name
        - rank (:obj:`int`): only rank == 0 can build, else return TextLogger that only save terminal output
    Returns:
        - logger (:obj:`TextLogger`): logger that save terminal output
        - tb_logger (:obj:`TensorBoardLogger` or :obj:`None`): logger that save output to tensorboard,
                                                              if rank != 0 then None
        - variable_record (:obj:`VariableRecord` or :obj:`None`): logger that record variable for further process,
                                                              if rank != 0 then None
    '''
    if cfg.common.type == 'rl':
        path = os.path.join(os.getcwd(), 'experiments', cfg.common.experiment_name, cfg.learner.player_id)
    else:
        path = os.path.join(os.getcwd(), 'experiments', cfg.common.experiment_name)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            print(f'{path} already exists!')
    # Note: Only support rank0 tb_logger, variable_record
    if rank == 0:
        logger = TextLogger(path, name=name)
        tb_logger = TensorBoardLogger(path, name=name)
        var_record_type = cfg.learner.get("var_record_type", None)
        if var_record_type is None:
            variable_record = VariableRecord(cfg.learner.hook.log_show.ext_args.freq)
        elif var_record_type == 'alphastar':
            variable_record = AlphaStarVarRecord(cfg.learner.hook.log_show.ext_args.freq)
        elif var_record_type == 'SupervisedStar':
            variable_record = SupervisedStarVarRecord(cfg.learner.hook.log_show.ext_args.freq)
        elif var_record_type == 'OriSupervisedStar':
            variable_record = OriSupervisedStarVarRecord(cfg.learner.hook.log_show.ext_args.freq)
        else:
            raise NotImplementedError("not support var_record_type: {}".format(var_record_type))
        return logger, tb_logger, variable_record
    else:
        logger = TextLogger(path, name=name)
        return logger, None, None


def build_logger_naive(path, name, level=logging.INFO, print_freq=1):
    r'''
    Overview:
        use config to build Textlogger and VariableRecord
    Arguments:
        - path (:obj:`str`): logger's save dir, please reference log_helper.TextLogger
        - name (:obj:`str`): the logger file name
        - level (:obj:`int` or :obj:`str`): Set the logging level of logger
        - rank (:obj:`int`): only rank == 0 can build, else return TextLogger that only save terminal output
    Returns:
        - logger (:obj:`TextLogger`): logger that save terminal output
        - variable_record (:obj:`VariableRecord`): logger that record variable for further process
    '''
    logger = TextLogger(path, name, level)
    variable_record = VariableRecord(print_freq)
    return logger, variable_record


def get_default_logger(name=None):
    r"""
    Overview:
        get the logger using logging.getLogger

    Arguments:
        - name (:obj:`str`): the name of logger, if None then get 'default_logger'

    Notes:
        you can reference Logger.manager.getLogger(name) in the python3 /logging/__init__.py
    """
    if name is None:
        name = 'default_logger'
    return logging.getLogger(name)


class TextLogger(object):
    r"""
    Overview:
        Logger that save terminal output to file

    Interface: __init__, info
    """

    def __init__(self, path, name=None, level=logging.INFO):
        r"""
        Overview:
            initialization method, create logger.
        Arguments:
            - path (:obj:`str`): logger's save dir
            - name (:obj:`str`): logger's name
            - level (:obj:`int` or :obj:`str`): Set the logging level of logger, reference Logger class setLevel method.
        """
        if name is None:
            name = 'default_logger'
        # ensure the path exists
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
        self.logger = self._create_logger(name, os.path.join(path, name + '.txt'), level=level)

    def _create_logger(self, name, path, level=logging.INFO):
        r"""
        Overview:
            create logger using logging
        Arguments:
            - name (:obj:`str`): logger's name
            - path (:obj:`str`): logger's save dir
        Returns:
            - (:obj`logger`): new logger
        """
        logger = logging.getLogger(name)
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format='[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
        if not logger.handlers:
            formatter = logging.Formatter('[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
            fh = logging.FileHandler(path, 'a')
            fh.setFormatter(formatter)
            logger.setLevel(level)
            logger.addHandler(fh)
            # handler = logging.StreamHandler()
            # handler.setFormatter(formatter)
            # logger.addHandler(handler)
            # logger.propagate = False
        return logger

    def info(self, s):
        r"""
        Overview:
            add message to logger
        Arguments:
            - s (:obj:`str`): message to add to logger
        Notes:
            you can reference Logger class in the python3 /logging/__init__.py
        """
        self.logger.info(s)

    def bug(self, s):
        r"""
        Overview:
            call logger.debug
        Arguments:
            - s (:obj:`str`): message to add to logger
        Notes:
            you can reference Logger class in the python3 /logging/__init__.py
        """
        self.logger.debug(s)

    def error(self, s):
        self.logger.error(s)


class TensorBoardLogger(object):
    r"""
    Overview:
        logger that save message to tensorboard

    Interface:
        __init__, add_scalar, add_text, add_scalars, add_histogram, add_figure, add_image, add_scalar_list,
        register_var, scalar_var_names
    """

    def __init__(self, path, name=None):
        r"""
        Overview:
            initialization method, create logger and set var names.
        Arguments:
            - path (:obj:`str`): logger save dir
            - name (:obj:`str`): logger name
        """
        if name is None:
            name = 'default_tb_logger'
        self.logger = SummaryWriter(os.path.join(path, name))  # get summary writer
        self._var_names = {
            'scalar': [],
            'text': [],
            'scalars': [],
            'histogram': [],
            'figure': [],
            'image': [],
        }

    def add_scalar(self, name, *args, **kwargs):
        r"""
        Overview:
            add message to scalar
        Arguments:
            - name (:obj:`str`): name to add which in self._var_names['scalar']
        """
        assert (name in self._var_names['scalar'])
        self.logger.add_scalar(name, *args, **kwargs)

    def add_text(self, name, *args, **kwargs):
        r"""
        Overview:
            add message to text
        Arguments:
            - name (:obj:`str`): name to add which in self._var_names['text']
        """
        assert (name in self._var_names['text'])
        self.logger.add_text(name, *args, **kwargs)

    def add_scalars(self, name, *args, **kwargs):
        r"""
        Overview:
            add message to scalars
        Arguments:
            - name (:obj:`str`): name to add which in self._var_names['scalars']
        """
        assert (name in self._var_names['scalars'])
        self.logger.add_scalars(name, *args, **kwargs)

    def add_histogram(self, name, *args, **kwargs):
        r"""
        Overview:
            add message to histogram
        Arguments:
            - name (:obj:`str`): name to add which in self._var_names['histogram']
        """
        assert (name in self._var_names['histogram'])
        self.logger.add_histogram(name, *args, **kwargs)

    def add_figure(self, name, *args, **kwargs):
        r"""
        Overview:
            add message to figure
        Arguments:
            - name (:obj:`str`): name to add which in self._var_names['figure']
        """
        assert (name in self._var_names['figure'])
        self.logger.add_figure(name, *args, **kwargs)

    def add_image(self, name, *args, **kwargs):
        r"""
        Overview:
            add message to image
        Arguments:
            - name (:obj:`str`): name to add which in self._var_names['image']
        """
        assert (name in self._var_names['image'])
        self.logger.add_image(name, *args, **kwargs)

    def add_val_list(self, val_list, viz_type):
        r"""
        Overview:
            add val_list info to tb
        Arguments:
            - val_list (:obj:`list`): include element(name, value, step) to be added
            - viz_type (:obs:`str`): must be in ['scalar', 'scalars', 'histogram']
        """
        assert (viz_type in ['scalar', 'scalars', 'histogram'])
        func_dict = {
            'scalar': self.add_scalar,
            'scalars': self.add_scalars,
            'histogram': self.add_histogram,
        }
        for n, v, s in val_list:
            func_dict[viz_type](n, v, s)

    def _no_contain_name(self, name):
        for k, v in self._var_names.items():
            if name in v:
                return False
        return True

    def register_var(self, name, var_type='scalar'):
        r"""
        Overview:
            add var to self_var._names

        Arguments:
            - name (:obj:`str`): name to add
            - var_type (:obj:`str`): the type of var to add to, defalut set to 'scalar',
                support [scalar', 'text', 'scalars', 'histogram', 'figure', 'image']
        """
        assert (var_type in self._var_names.keys())
        # assert (self._no_contain_name(name))
        self._var_names[var_type].append(name)

    @property
    def scalar_var_names(self):
        r"""
        Overview:
            return scalar_var_names
        Returns:
            - names(:obj:`list` of :obj:`str`): self._var_names['scalar']
        """
        return self._var_names['scalar']


class VariableRecord(object):
    r"""
    Overview:
        logger that record variable for further process

    Interface:
        __init__, register_var, update_var, get_var_names, get_var_text, get_vars_tb_format, get_vars_text
    """

    def __init__(self, length):
        r"""
        Overview:
            init the VariableRecord
        Arguments:
            - length (:obj:`int`): the length to average across, if less than 10 then will be set to 10
        """
        self.var_dict = {'scalar': {}}
        self.length = max(length, 10)  # at least average across 10 iteration

    def register_var(self, name, length=None, var_type='scalar'):
        r"""
        Overview:
            add var to self_var._names, calculate it's average value
        Arguments:
            - name (:obj:`str`): name to add
            - length (:obj:`int` or :obj:`None`): length of iters to average, default set to self.length
            - var_type (:obj:`str`): the type of var to add to, defalut set to 'scalar'
        """
        assert (var_type in ['scalar'])
        lens = self.length if length is None else length
        self.var_dict[var_type][name] = AverageMeter(lens)

    def update_var(self, info):
        r"""
        Overview:
            update vars
        Arguments:
            - info (:obj:`dict`): key is var type and value is the corresponding variable name
        """
        assert isinstance(info, dict)
        for k, v in info.items():
            var_type = self._get_var_type(k)
            self.var_dict[var_type][k].update(v)

    def _get_var_type(self, k):
        for var_type, var_type_dict in self.var_dict.items():
            if k in var_type_dict.keys():
                return var_type
        raise KeyError("invalid key({}) in variable record".format(k))

    def get_var_names(self, var_type='scalar'):
        r"""
        Overview:
            get the corresponding variable names of a certain var_type
        Arguments:
            - var_type (:obj:`str`): defalut set to 'scalar', support [scalar']
        Returns:
            - keys (:obj:`list` of :obj:`str`): the var names of a certain var_type
        """
        return self.var_dict[var_type].keys()

    def get_var_text(self, name, var_type='scalar'):
        r"""
        Overview:
            get the text discroption of var
        Arguments:
            - name (:obj:`str`): name of the var to query
            - var_type(:obj:`str`): default set to scalar, support ['scalar']
        Returns:
            - text(:obj:`str`): the corresponding text description
        """
        assert (var_type in ['scalar'])
        if var_type == 'scalar':
            handle_var = self.var_dict[var_type][name]
            return '{}: val({:.6f})|avg({:.6f})'.format(name, handle_var.val, handle_var.avg)
        else:
            raise NotImplementedError

    def get_vars_tb_format(self, keys, cur_step, var_type='scalar', **kwargs):
        r"""
        Overview:
            get the tb_format description of var
        Arguments:
            - keys (:obj:`list` of :obj:`str`): keys(names) of the var to query
            - cur_step (:obj:`int`): the current step
            - var_type(:obj:`str`): default set to scalar, support support ['scalar']
        Returns:
            - ret (:obj:`list` of :obj:`list` of :obj:`str`): the list of tb_format info of vars queried
        """
        assert (var_type in ['scalar'])
        if var_type == 'scalar':
            ret = []
            var_keys = self.get_var_names(var_type)
            for k in keys:
                if k in var_keys:
                    v = self.var_dict[var_type][k]
                    if k == 'grad':
                        ret.append([k, v.val, cur_step])
                    else:
                        ret.append([k, v.avg, cur_step])
            return ret
        else:
            raise NotImplementedError

    def get_vars_text(self):
        r"""
        Overview:
            get the string description of var
        Returns:
            - ret (:obj:`list` of :obj:`str`): the list of text description of vars queried
        """
        headers = ["Name", "Value", "Avg"]
        data = []
        for k in self.get_var_names('scalar'):
            handle_var = self.var_dict['scalar'][k]
            data.append([k, "{:.6f}".format(handle_var.val), "{:.6f}".format(handle_var.avg)])
        s = "\n" + tabulate(data, headers=headers, tablefmt='grid')
        return s


class LogDict(dict):

    def _transform(self, data):
        if isinstance(data, torch.Tensor):
            if data.shape == (1,) or data.shape == ():
                new_data = data.item()
            else:
                new_data = data.tolist()
        else:
            new_data = data
        return new_data

    def __setitem__(self, key, value):
        new_value = self._transform(value)
        super().__setitem__(key, new_value)

    def update(self, data):
        for k, v in data.items():
            self.__setitem__(k, v)


class LogDictList(defaultdict):

    def _transform(self, data):
        if isinstance(data, torch.Tensor):
            if data.shape == (1,) or data.shape == ():
                new_data = data.item()
            else:
                new_data = data.tolist()
        else:
            new_data = data
        return new_data

    def add(self, key, value):
        new_value = self._transform(value)
        self[key].append(new_value)


def build_log_buffer():
    r"""
    Overview:
        builg log buffer, a subclass of dict which can transform the input data into log format
    Returns:
        - log_buffer (:obj:`LogDict`): log buffer dict
    """
    return LogDict()

class MoveAverageMeter(object):
    def __init__(self,length=1000):
        self.length = length
        self.reset()
    def reset(self):
        self.history = deque(maxlen=self.length)
        self._val = 0.0
        self._count = 0

    def update(self, val):
        if isinstance(val, torch.Tensor):
            val = val.item()
        assert (isinstance(val, list) or isinstance(val, numbers.Integral) or isinstance(val, numbers.Real))
        self._count += 1
        s1 = len(self.history)
        if s1 < self.length:
            self._val = (1-1.0/(s1+1))*self._val+1.0/(s1+1)*val
            self.history.append(val)
        else:
            left_val = self.history.popleft()
            self._val = self._val+(val-left_val)/self.length
            self.history.append(val)

    @property
    def val(self):
        return self._val

    @property
    def count(self):
        return self._count

    @val.setter
    def val(self):
        raise NotImplementedError

    @count.setter
    def count(self):
        raise NotImplementedError



class AverageMeter(object):
    r"""
    Overview:
        Computes and stores the average and current value, scalar and 1D-array
    Interface:
        __init__, reset, update
    """

    def __init__(self, length=0):
        r"""
        Overview:
            init AverageMeter class
        Arguments:
            - length (:obj:`int`) : set the default length of iters to average
        """
        assert (length > 0)
        self.length = length
        self.reset()

    def reset(self):
        r"""
        Overview:
            reset AverageMeter class
        """
        self.history = []
        self.val = 0.0
        self.avg = 0.0

    def update(self, val):
        r"""
        Overview:
            update AverageMeter class, append the val to the history and calculate the average
        Arguments:
            - val (:obj:`numbers.Integral` or :obj:`list` or :obj:`numbers.Real` ) : the latest value
        """
        if isinstance(val, torch.Tensor):
            val = val.item()
        assert (isinstance(val, list) or isinstance(val, numbers.Integral) or isinstance(val, numbers.Real))
        self.history.append(val)
        if len(self.history) > self.length:
            del self.history[0]

        self.val = self.history[-1]
        self.avg = np.mean(self.history, axis=0)


class EmaMeter(object):
    def __init__(self, decay, warm_up_size):
        assert (warm_up_size > 0)
        assert 0 <= decay <= 1
        self._decay = decay
        self._warm_up_size = warm_up_size
        self.reset()

    def reset(self):
        self._count = 0
        self._val = 0

    def update(self, val):
        if isinstance(val, torch.Tensor):
            val = val.item()
        assert (isinstance(val, list) or isinstance(val, numbers.Integral) or isinstance(val, numbers.Real))
        if self._count >= self._warm_up_size:
            self._val = self._decay * self._val + (1 - self._decay) * val
        else:
            self._val = (self._val * self._count + val) / (self._count + 1)
        self._count += 1

    @property
    def val(self):
        return self._val

    @property
    def count(self):
        return self._count

    @val.setter
    def val(self):
        raise NotImplementedError

    @count.setter
    def count(self):
        raise NotImplementedError


class DistributionTimeImage(object):
    r"""
    Overview:
        DistributionTimeImage can be used to store images accorrding to time_steps,
        for data with 3 dims(time, category, value)
    Interface:
        __init__, add_one_time_step, get_image
    """

    def __init__(self, maxlen=600, val_range=None):
        r"""
        Overview:
            init the DistributionTimeImage class
        Arguments:
            - maxlen (:obj:`int`): the max length of data inputs
            - val_range (:obj:`dict` or :obj:`None`): contain :obj:`int` type val_range['min'] and val_range['max'],
                                                      default set to None
        """
        self.maxlen = maxlen
        self.val_range = val_range
        self.img = np.ones((maxlen, maxlen))
        self.time_step = 0
        self.one_img = np.ones((maxlen, maxlen))

    def add_one_time_step(self, data):
        r"""
        Overview:
            step one timestep in DistributionTimeImage and add the data to distribution image
        Arguments:
            - data (:obj:`np.array`):the data input
        """
        assert (isinstance(data, np.ndarray))
        data = np.expand_dims(data, 1)
        data = cv2.resize(data, (1, self.maxlen), interpolation=cv2.INTER_LINEAR)
        if self.time_step >= self.maxlen:
            self.img = np.concatenate([self.img[:, 1:], data])
        else:
            self.img[:, self.time_step:self.time_step + 1] = data
            self.time_step += 1

    def get_image(self):
        r"""
        Overview:
            return the distribution image
        Returns:
            img (:obj:`bp.ndarray`): the calculated distribution image
        """
        norm_img = np.copy(self.img)
        valid = norm_img[:, :self.time_step]
        if self.val_range is None:
            valid = (valid - valid.min()) / (valid.max() - valid.min())
        else:
            valid = np.clip(valid, self.val_range['min'], self.val_range['max'])
            valid = (valid - self.val_range['min']) / (self.val_range['max'] - self.val_range['min'])
        norm_img[:, :self.time_step] = valid
        return np.stack([self.one_img, norm_img, norm_img], axis=0)


def pretty_print(result, direct_print=True):
    r"""
    Overview:
        print the result in a pretty way
    Arguments:
        - result (:obj:`dict`): the result to print
        - direct_print (:obj:`bool`): whether to print directly
    Returns:
        - string (:obj:`str`): the printed result in str format
    """
    result = result.copy()
    out = {}
    for k, v in result.items():
        if v is not None:
            out[k] = v
    cleaned = json.dumps(out)
    string = yaml.safe_dump(json.loads(cleaned), default_flow_style=False)
    if direct_print:
        print(string)
    return string


class AlphaStarVarRecord(VariableRecord):
    def __init__(self, length):
        super(AlphaStarVarRecord, self).__init__(length)
        self.keys_mapping = {
            'winloss': 'winloss',
            'bo': 'build_order',
            'bu': 'built_unit',
            'effect': 'effect',
            'upgrade': 'upgrade',
            'battle': 'battle',
            'upgo': 'upgo',
            'kl': 'kl',
            'ent': 'entropy',

            'rew': 'reward',
            'val': 'value',
            'td': 'td',
            'at': 'action_type',
            'delay': 'delay',
            'queued': 'queued',
            'su': 'selected_units',
            'tu': 'target_unit',
            'tl': 'target_location',
            'total': 'total',
            'name': 'name'

        }

    def get_vars_text(self):
        headers = ['name', 'val', 'name', 'rew', 'val', 'td', 'at', 'delay', 'queued', 'su', 'tu', 'tl', 'total']

        data = []
        k0 = ['cur_lr', 'data_time', 'train_time', 'forward_time', 'backward_time', 'total_loss', 'gradient',
              'kl/extra_at']
        k1 = ['winloss', 'bo', 'bu', 'effect', 'upgrade', 'battle', 'upgo', 'kl', 'ent']
        k2 = ['rew', 'val', 'td', 'at', 'delay', 'queued', 'su', 'tu', 'tl', 'total']

        all_vars = self.get_var_names('scalar')
        for i in range(max(len(k1), len(k0))):
            d = []
            if i < len(k0):
                if k0[i] == 'gradient':
                    d += [k0[i], "{:.6f}".format(self.var_dict['scalar'][k0[i]].val)]
                else:
                    d += [k0[i], "{:.6f}".format(self.var_dict['scalar'][k0[i]].avg)]
            else:
                d += ['', '']

            if i < len(k1):
                d += [k1[i]]
                vals = []
                for k in k2:
                    var_key = self.keys_mapping[k1[i]] + '/' + self.keys_mapping[k]
                    if var_key in all_vars:
                        vals.append("{:.6f}".format(self.var_dict['scalar'][var_key].avg))
                    else:
                        vals.append('')
                d += vals
            data.append(d)
        s = "\n" + tabulate(data, headers=headers, tablefmt='grid')
        return s


class SupervisedStarVarRecord(VariableRecord):
    def get_vars_text(self):
        # short_dict = {
        #     'at':'action_type',
        #     'delay':'delay',
        #     'queued':'queued',
        #     'su':'selected_units',
        #     'tu':'target_unit',
        #     'tl':'target_location',
        # }
        # # table_header = ['name','at', 'delay', 'queued', 'su', 'tu', 'tl' ]

        table_header = ['name', 'cur_lr', 'data_time', 'train_time', 'forward_time', 'backward_time', 'total_loss',
                        'grad_norm']
        table_data = []

        for row in ['value', 'name', 'loss', 'acc']:
            if row == 'value':
                row_data = ['value']
                for col in table_header[1:]:
                    row_data.append("{:.6f}".format(self.var_dict['scalar'][col].val))
            elif row == 'name':
                row_data = ['name', 'action_type', 'delay', 'queued', 'selected_units', 'target_unit',
                            'target_location', 'grad_clip_value']
            else:
                row_data = [row]
                for col in ['action_type', 'delay', 'queued', 'selected_units', 'target_unit', 'target_location']:
                    row_data.append("{:.6f}".format(self.var_dict['scalar'][col + '_' + row].val))
                if row == 'loss':
                    row_data.append(self.var_dict['scalar']['grad_clip_value'].val)
                else:
                    row_data.append('')
            table_data.append(row_data)
        s = "\n" + (tabulate(table_data, headers=table_header, tablefmt='grid'))
        return s

    class SupervisedStarVarRecord(VariableRecord):
        def get_vars_text(self):
            # short_dict = {
            #     'at':'action_type',
            #     'delay':'delay',
            #     'queued':'queued',
            #     'su':'selected_units',
            #     'tu':'target_unit',
            #     'tl':'target_location',
            # }
            # # table_header = ['name','at', 'delay', 'queued', 'su', 'tu', 'tl' ]

            table_header = ['name', 'cur_lr', 'data_time', 'train_time', 'forward_time', 'backward_time', 'total_loss',
                            'grad_norm']
            table_data = []

            for row in ['value', 'name', 'loss', 'acc']:
                if row == 'value':
                    row_data = ['value']
                    for col in table_header[1:]:
                        row_data.append("{:.6f}".format(self.var_dict['scalar'][col].val))
                elif row == 'name':
                    row_data = ['name', 'action_type', 'delay', 'queued', 'selected_units', 'target_unit',
                                'target_location', 'grad_clip_value']
                else:
                    row_data = [row]
                    for col in ['action_type', 'delay', 'queued', 'selected_units', 'target_unit', 'target_location']:
                        row_data.append("{:.6f}".format(self.var_dict['scalar'][col + '_' + row].val))
                    if row == 'loss':
                        row_data.append(self.var_dict['scalar']['grad_clip_value'].val)
                    else:
                        row_data.append('')
                table_data.append(row_data)
            s = "\n" + (tabulate(table_data, headers=table_header, tablefmt='grid'))
            return s


class OriSupervisedStarVarRecord(VariableRecord):
    # overwrite
    def register_var(self, name, length=None, var_type='scalar', var_item_keys=None):
        assert (var_type in ['scalar', '1darray'])
        lens = self.length if length is None else length
        self.var_dict[var_type][name] = AverageMeter(lens)
        if not hasattr(self, 'var_item_keys'):
            self.var_item_keys = {}
        self.var_item_keys[name] = var_item_keys

    # overwrite
    def _get_vars_text_1darray(self):
        s = "\n"
        for k in self.get_var_names('1darray'):
            val = self.var_dict['1darray'][k].avg
            if k == 'action_type':
                s += '{}:\t'.format(k)
                items = [[n, v] for n, v in zip(self.var_item_keys[k], val) if v > 0]
                items = sorted(items, key=lambda x: float(x[1]), reverse=True)
                for n, v in items:
                    s += '{}({:.2f})  '.format(n, v)
                s += '\n'
            elif k == 'all_acc':
                s += '{}:\t'.format(k)
                items = [(idx, acc) for idx, acc in enumerate(val) if acc > 0]
                items = sorted(items, key=lambda x: float(x[1]), reverse=True)
                for n, v in items:
                    s += '{}({:.2f})  '.format(ACTION_INFO_MASK[ACTIONS_REORDER_INV[n]]['name'], v)
                s += '\n'
            else:
                s += '{}:\t'.format(k)
                for n, v in zip(self.var_item_keys[k], val):
                    s += '{}({:.2f})  '.format(n, v)
                s += '\n'
        return s

    # overwrite
    def _get_vars_tb_format_1darray(self, keys, cur_step, viz_type=None):
        assert (viz_type in ['scalars', 'histogram'])
        if viz_type == 'scalars':
            ret = []
            var_keys = self.get_var_names('1darray')
            for k in keys:
                if k in var_keys:
                    v = self.var_dict['1darray'][k]
                    scalars = {k: v for k, v in zip(self.var_item_keys[k], v.avg)}
                    ret.append([k, scalars, cur_step])
            return ret
        elif viz_type == 'histogram':
            ret = []
            var_keys = self.get_var_names('1darray')
            for k in keys:
                if k in var_keys:
                    v = self.var_dict['1darray'][k]
                    ret.append([k, v.avg, cur_step])
            return ret
