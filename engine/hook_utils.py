# -*- coding: utf-8 -*-
"""
Hook 工具模块

提供 stderr 日志过滤和模型卸载钩子功能。
集成 ComfyUI 的模型卸载钩子，确保 llama.cpp 模型与 ComfyUI 共用显存时正确清理。
支持多级别日志过滤和上下文管理器模式。

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""

import io
import sys
import re
import contextlib
import functools


class StderrFilter:
    """
    Stderr 输出过滤器

    过滤 llama.cpp 内部的 find_slot 等冗余日志，
    减少日志 IO 开销对推理性能的影响。
    """

    def __init__(self):
        self.original_stderr = sys.stderr
        self.buffer = io.StringIO()

    def write(self, text):
        if "find_slot" in text:
            match = re.search(
                r'find_slot: non-consecutive token position (\d+) after (\d+) for sequence (\d+) with (\d+) new tokens',
                text
            )
            if match:
                pos = match.group(1)
                prev_pos = match.group(2)
                seq = match.group(3)
                tokens = match.group(4)
                translated = f"find_slot: 序列 {seq} 的非连续token位置 {pos}（前一位置 {prev_pos}），使用 {tokens} 个新tokens"
                self.original_stderr.write(translated + "\n")
            else:
                translated = text.replace("find_slot: non-consecutive token position", "find_slot: 非连续token位置")
                translated = translated.replace("after", "在")
                translated = translated.replace("for sequence", "对于序列")
                translated = translated.replace("with", "使用")
                translated = translated.replace("new tokens", "新tokens")
                self.original_stderr.write(translated)
        else:
            self.original_stderr.write(text)

    def flush(self):
        self.original_stderr.flush()


@contextlib.contextmanager
def filter_stderr():
    """
    Stderr 过滤上下文管理器

    用法:
        with filter_stderr():
            llm.create_chat_completion(...)
    """
    filter = StderrFilter()
    sys.stderr = filter
    try:
        yield
    finally:
        sys.stderr = filter.original_stderr


class MultiLevelFilter:
    """
    多级别日志过滤器

    支持不同级别的日志过滤：
    - DEBUG: 调试信息
    - INFO: 一般信息
    - WARNING: 警告信息
    - ERROR: 错误信息
    """

    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

    def __init__(self, level=INFO):
        self.original_stderr = sys.stderr
        self.level = level
        self._filter_patterns = [
            (r"find_slot", self.INFO),
            (r"kv cache", self.DEBUG),
            (r"loading model", self.INFO),
            (r"error|failed|exception", self.ERROR),
        ]

    def write(self, text):
        if self.level <= self.DEBUG:
            self.original_stderr.write(text)
            return

        should_print = True
        for pattern, min_level in self._filter_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                if self._get_level(pattern) < self.level:
                    should_print = False
                    break

        if should_print:
            self.original_stderr.write(text)

    def _get_level(self, pattern):
        for p, lvl in self._filter_patterns:
            if p == pattern:
                return lvl
        return self.INFO

    def flush(self):
        self.original_stderr.flush()

    def set_level(self, level):
        self.level = level

    def add_filter(self, pattern, level):
        self._filter_patterns.append((pattern, level))


@contextlib.contextmanager
def filtered_stderr(level=MultiLevelFilter.INFO):
    """
    多级别日志过滤上下文管理器

    用法:
        with filtered_stderr(MultiLevelFilter.WARNING):
            llm.create_chat_completion(...)
    """
    filter = MultiLevelFilter(level)
    sys.stderr = filter
    try:
        yield
    finally:
        sys.stderr = filter.original_stderr


try:
    import comfy.model_management as mm
    _has_comfy = True
except ImportError:
    _has_comfy = False
    mm = None


class ModelUnloadHook:
    """
    模型卸载钩子管理器

    当 ComfyUI 调用 unload_all_models 时，自动清理 llama.cpp 模型，
    避免显存泄漏。
    """

    _instance = None
    _hooked = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._llama_storage = None
        self._original_unload = None

    def register_hook(self, llama_storage):
        """
        注册钩子

        Args:
            llama_storage: LLAMA_CPP_STORAGE 类，用于获取 clean 方法
        """
        if not _has_comfy:
            return

        self._llama_storage = llama_storage

        if not ModelUnloadHook._hooked:
            self._install_hook()
            ModelUnloadHook._hooked = True

    def _install_hook(self):
        """
        安装模型卸载钩子
        """
        if not _has_comfy or not hasattr(mm, "unload_all_models"):
            return

        if hasattr(mm, "unload_all_models_backup"):
            return

        self._original_unload = mm.unload_all_models

        @functools.wraps(self._original_unload)
        def patched_unload_all_models(*args, **kwargs):
            if self._llama_storage is not None:
                try:
                    self._llama_storage.clean(all=True)
                    print("[Acceleration] llama.cpp 模型已通过卸载钩子清理")
                except Exception:
                    pass

            result = self._original_unload(*args, **kwargs)
            return result

        mm.unload_all_models = patched_unload_all_models
        mm.unload_all_models_backup = self._original_unload
        print("[Acceleration] 模型卸载钩子已应用！")

    def uninstall_hook(self):
        """
        卸载钩子，恢复原始函数
        """
        if not _has_comfy or self._original_unload is None:
            return

        if hasattr(mm, "unload_all_models_backup"):
            mm.unload_all_models = mm.unload_all_models_backup
            delattr(mm, "unload_all_models_backup")
            ModelUnloadHook._hooked = False
            print("[Acceleration] 模型卸载钩子已移除")


def install_model_unload_hook(llama_storage):
    """
    安装模型卸载钩子的便捷函数

    Args:
        llama_storage: LLAMA_CPP_STORAGE 类

    Returns:
        ModelUnloadHook: 钩子管理器实例
    """
    hook_manager = ModelUnloadHook()
    hook_manager.register_hook(llama_storage)
    return hook_manager


def uninstall_model_unload_hook():
    """
    卸载模型卸载钩子的便捷函数
    """
    hook_manager = ModelUnloadHook()
    hook_manager.uninstall_hook()


def apply_acceleration_hooks(llama_storage):
    """
    应用所有加速钩子的便捷函数

    Args:
        llama_storage: LLAMA_CPP_STORAGE 类

    Returns:
        ModelUnloadHook: 卸载钩子管理器
    """
    return install_model_unload_hook(llama_storage)
