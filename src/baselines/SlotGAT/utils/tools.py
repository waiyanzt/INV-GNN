"""Utility functions extracted from SlotGAT_ICML23."""

import os
import random
import json
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def func_args_parse(*args, **kargs):
    return args, kargs


def single_feat_net(net, *args, **kargs):
    return net(*args, **kargs)


class blank_profile():
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self, *args, **kwargs):
        return self
    def __exit__(self, *args, **kwargs):
        pass
    def step(self):
        pass


class vis_data_collector():
    """Collects data for visualization / logging during training."""
    def __init__(self):
        self.data_dict = {}
        self.tensor_dict = {}

    def save_meta(self, meta_data, meta_name):
        self.data_dict["meta"] = {meta_name: meta_data}
        self.data_dict["meta"]["tensor_names"] = []

    def collect_in_training(self, data, name, re, epoch, r=4):
        if f"re-{re}" not in self.data_dict.keys():
            self.data_dict[f"re-{re}"] = {}
        if f"epoch-{epoch}" not in self.data_dict[f"re-{re}"].keys():
            self.data_dict[f"re-{re}"][f"epoch-{epoch}"] = {}
        if type(data) == float:
            data = round(data, r)
        self.data_dict[f"re-{re}"][f"epoch-{epoch}"][name] = data

    def collect_in_run(self, data, name, re, r=4):
        if f"re-{re}" not in self.data_dict.keys():
            self.data_dict[f"re-{re}"] = {}
        if type(data) == float:
            data = round(data, r)
        self.data_dict[f"re-{re}"][name] = data

    def collect_whole_process(self, data, name):
        self.data_dict[name] = data

    def collect_whole_process_tensor(self, data, name):
        self.tensor_dict[name] = data
        self.data_dict["meta"]["tensor_names"].append(name)

    def save(self, fn):
        f = open(fn + ".json", 'w')
        json.dump(self.data_dict, f, indent=4)
        f.close()
        for k, v in self.tensor_dict.items():
            torch.save(v, fn + "_" + k + ".pt")

    def load(self, fn):
        f = open(fn, 'r')
        self.data_dict = json.load(f)
        f.close()
        for name in self.data_dict["meta"]["tensor_names"]:
            self.tensor_dict[name] = torch.load(name + ".pt")

    def trans_to_numpy(self, name, epoch_range=None):
        data = []
        re = 0
        while f"re-{re}" in self.data_dict.keys():
            data.append([])
            for i in range(epoch_range):
                data[re].append(self.data_dict[f"re-{re}"][f"epoch-{i}"][name])
            re += 1
        data = np.array(data)
        return np.mean(data, axis=0), np.std(data, axis=0)
