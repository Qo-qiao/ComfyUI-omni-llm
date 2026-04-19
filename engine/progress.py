# -*- coding: utf-8 -*-
"""
进度条模块

封装 comfy.utils.ProgressBar，提供统一的进度条显示接口。
支持与 tqdm 兼容的 API 设计。

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""

import sys


try:
    import comfy.utils
    _has_comfy = True
except ImportError:
    _has_comfy = False
    comfy = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class cqdm:
    """
    进度条封装类，集成 ComfyUI 原生进度条和 tqdm

    提供与 tqdm 兼容的接口，同时支持 ComfyUI 的 ProgressBar 显示。
    """

    def __init__(self, iterable=None, total=None, desc="Processing", disable=False, **kwargs):
        self.iterable = iterable
        self.total = total
        self.desc = desc

        if iterable is not None and total is None:
            try:
                self.total = len(iterable)
            except (TypeError, AttributeError):
                self.total = None

        if _has_comfy and comfy is not None:
            self.pbar = comfy.utils.ProgressBar(self.total) if self.total is not None else None
        else:
            self.pbar = None

        if tqdm is not None:
            self.tqdm = tqdm(
                iterable=self.iterable,
                total=self.total,
                desc=self.desc,
                disable=disable,
                dynamic_ncols=True,
                file=sys.stdout,
                **kwargs
            )
        else:
            self.tqdm = None

    def __iter__(self):
        if self.tqdm is None:
            return
        for item in self.tqdm:
            if self.pbar:
                self.pbar.update(1)
            yield item

    def update(self, n=1):
        if self.tqdm:
            self.tqdm.update(n)
        if self.pbar:
            self.pbar.update(n)

    def set_description(self, desc):
        if self.tqdm:
            self.tqdm.set_description(desc)

    def set_postfix(self, *args, **kwargs):
        if self.tqdm:
            self.tqdm.set_postfix(*args, **kwargs)

    def close(self):
        if self.tqdm is not None:
            self.tqdm.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        return self.total


def create_progress_bar(total=None, desc="Processing", disable=False):
    """
    创建进度条的工厂函数

    Args:
        total: 总进度数
        desc: 进度条描述
        disable: 是否禁用进度条

    Returns:
        cqdm: 进度条实例
    """
    return cqdm(total=total, desc=desc, disable=disable)
