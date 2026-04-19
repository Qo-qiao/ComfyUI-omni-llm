# -*- coding: utf-8 -*-
"""
TurboQuant KV Cache 压缩模块

TurboQuant: 谷歌用于压缩LLM键值缓存的向量量化算法
Reference: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026)

Two-stage vector quantization:
Stage 1 (MSE): Random rotation + per-coordinate Lloyd-Max quantization
Stage 2 (QJL): 1-bit Quantized Johnson-Lindenstrauss on residuals for unbiased inner products

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict, List, Any
from functools import lru_cache


class LloydMaxCodebook:
    """
    Lloyd-Max optimal scalar quantizer for the Beta distribution
    arising from random rotation of unit-norm vectors.
    """
    
    def __init__(self, d: int, bits: int):
        self.d = d
        self.bits = bits
        self.n_levels = 2 ** bits
        self.sigma = 1.0 / math.sqrt(d)
        self.centroids, self.boundaries = self._solve_codebook()
    
    def _solve_codebook(self):
        try:
            from scipy import integrate
            use_scipy = True
        except ImportError:
            use_scipy = False
        
        n_levels = self.n_levels
        sigma = self.sigma
        
        if use_scipy:
            def pdf(x):
                return (1.0 / math.sqrt(2 * math.pi * sigma ** 2)) * math.exp(-x * x / (2 * sigma ** 2))
            
            lo, hi = -3.5 * sigma, 3.5 * sigma
            centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]
            
            for _ in range(200):
                boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
                edges = [lo * 3] + boundaries + [hi * 3]
                new_centroids = []
                for i in range(n_levels):
                    a, b = edges[i], edges[i + 1]
                    num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
                    den, _ = integrate.quad(pdf, a, b)
                    new_centroids.append(num / den if den > 1e-15 else centroids[i])
                
                if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < 1e-10:
                    break
                centroids = new_centroids
        else:
            lo, hi = -3.5 * sigma, 3.5 * sigma
            step = (hi - lo) / n_levels
            centroids = [lo + step * (i + 0.5) for i in range(n_levels)]
            boundaries = [lo + step * (i + 1) for i in range(n_levels - 1)]
        
        return torch.tensor(centroids, dtype=torch.float32), torch.tensor(boundaries, dtype=torch.float32)


def generate_rotation_matrix(d: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """Generate a random orthogonal rotation matrix via QR decomposition of Gaussian matrix."""
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


def generate_qjl_matrix(d: int, m: Optional[int] = None, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """Generate the random projection matrix S for QJL."""
    if m is None:
        m = d
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    S = torch.randn(m, d, generator=gen)
    return S.to(device)


class TurboQuantMSE(nn.Module):
    """
    Stage 1: MSE-optimal quantizer.
    Randomly rotates, then applies per-coordinate Lloyd-Max quantization.
    """
    
    def __init__(self, d: int, bits: int, seed: int = 42, device: str = "cpu"):
        super().__init__()
        self.d = d
        self.bits = bits
        self.device = device
        
        self.register_buffer("Pi", generate_rotation_matrix(d, seed=seed, device=device))
        self.codebook = LloydMaxCodebook(d, bits)
        self.register_buffer("centroids", self.codebook.centroids.to(device))
        self.register_buffer("boundaries", self.codebook.boundaries.to(device))
    
    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random rotation: y = Pi @ x."""
        return x @ self.Pi.T
    
    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """Undo rotation: x = Pi^T @ y."""
        return y @ self.Pi
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize vectors to codebook indices."""
        y = self.rotate(x)
        diffs = y.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1)
        return indices
    
    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Dequantize indices back to vectors."""
        y_hat = self.centroids[indices]
        return self.unrotate(y_hat)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full quantize-dequantize cycle. Returns: (reconstructed_x, indices)"""
        indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        return x_hat, indices


class TurboQuantProd(nn.Module):
    """
    Stage 1 + Stage 2: Unbiased inner product quantizer.
    Uses (b-1)-bit MSE quantizer + 1-bit QJL on residuals.
    """
    
    def __init__(self, d: int, bits: int, qjl_dim: Optional[int] = None, seed: int = 42, device: str = "cpu"):
        super().__init__()
        self.d = d
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.qjl_dim = qjl_dim or d
        self.device = device
        
        self.mse = TurboQuantMSE(d, self.mse_bits, seed=seed, device=device)
        self.register_buffer("S", generate_qjl_matrix(d, m=self.qjl_dim, seed=seed + 1, device=device))
    
    def quantize(self, x: torch.Tensor) -> dict:
        """Full TurboQuant quantization."""
        x_hat, mse_indices = self.mse(x)
        residual = x - x_hat
        residual_norm = torch.norm(residual, dim=-1, keepdim=True)
        
        projected = residual @ self.S.T
        qjl_signs = torch.sign(projected)
        qjl_signs[qjl_signs == 0] = 1.0
        
        return {
            "mse_indices": mse_indices,
            "qjl_signs": qjl_signs,
            "residual_norm": residual_norm.squeeze(-1),
        }
    
    def dequantize(self, compressed: dict) -> torch.Tensor:
        """Dequantize MSE component."""
        return self.mse.dequantize(compressed["mse_indices"])
    
    def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
        """Compute unbiased inner product estimate: <y, x> using compressed representation."""
        x_mse = self.mse.dequantize(compressed["mse_indices"])
        term1 = (y * x_mse).sum(dim=-1)
        
        y_projected = y @ self.S.T
        qjl_ip = (y_projected * compressed["qjl_signs"]).sum(dim=-1)
        
        m = self.qjl_dim
        correction_scale = math.sqrt(math.pi / 2) / m
        term2 = compressed["residual_norm"] * correction_scale * qjl_ip
        
        return term1 + term2
    
    def forward(self, x: torch.Tensor) -> dict:
        """Quantize input vectors."""
        return self.quantize(x)


class TurboQuantKVCache:
    """
    KV cache wrapper using TurboQuant to compress keys and values.
    Drop-in replacement for standard KV cache.
    """
    
    def __init__(self, d_key: int, d_value: int, bits: int = 3, seed: int = 42, device: str = "cpu"):
        self.d_key = d_key
        self.d_value = d_value
        self.bits = bits
        self.device = device
        
        self.key_quantizer = TurboQuantProd(d_key, bits, seed=seed, device=device)
        self.value_quantizer = TurboQuantMSE(d_value, bits, seed=seed + 100, device=device)
        
        self.key_cache = []
        self.value_cache = []
    
    def append(self, keys: torch.Tensor, values: torch.Tensor):
        """Append new key-value pairs to cache."""
        orig_shape = keys.shape
        flat_keys = keys.reshape(-1, self.d_key)
        flat_values = values.reshape(-1, self.d_value)
        
        compressed_keys = self.key_quantizer.quantize(flat_keys)
        value_indices = self.value_quantizer.quantize(flat_values)
        
        self.key_cache.append({
            "mse_indices": compressed_keys["mse_indices"],
            "qjl_signs": compressed_keys["qjl_signs"],
            "residual_norm": compressed_keys["residual_norm"],
            "shape": orig_shape,
        })
        self.value_cache.append({
            "indices": value_indices,
            "shape": values.shape,
        })
    
    def attention_scores(self, queries: torch.Tensor) -> torch.Tensor:
        """Compute attention scores using unbiased inner product estimation."""
        scores = []
        for cached in self.key_cache:
            s = self.key_quantizer.inner_product(queries, cached)
            scores.append(s)
        return torch.cat(scores, dim=-1) if scores else torch.tensor([])
    
    def get_values(self) -> torch.Tensor:
        """Reconstruct all cached values."""
        values = []
        for cached in self.value_cache:
            v = self.value_quantizer.dequantize(cached["indices"])
            values.append(v)
        return torch.cat(values, dim=0) if values else torch.tensor([])
    
    def memory_usage_bits(self) -> dict:
        """Estimate memory usage in bits."""
        n_keys = sum(c["mse_indices"].numel() for c in self.key_cache) if self.key_cache else 0
        n_qjl = sum(c["qjl_signs"].numel() for c in self.key_cache) if self.key_cache else 0
        n_norms = sum(c["residual_norm"].numel() for c in self.key_cache) if self.key_cache else 0
        n_values = sum(c["indices"].numel() for c in self.value_cache) if self.value_cache else 0
        
        key_bits = n_keys * self.key_quantizer.mse_bits + n_qjl * 1 + n_norms * 16
        value_bits = n_values * self.bits
        fp16_equivalent = (n_keys + n_values) * 16
        
        return {
            "key_bits": key_bits,
            "value_bits": value_bits,
            "total_bits": key_bits + value_bits,
            "fp16_bits": fp16_equivalent,
            "compression_ratio": fp16_equivalent / (key_bits + value_bits) if (key_bits + value_bits) > 0 else 0,
        }
    
    def __len__(self):
        return sum(c["mse_indices"].shape[0] for c in self.key_cache) if self.key_cache else 0


class TurboQuantCompressorV2:
    """
    Compressor for direct asymmetric attention computation.
    Computes attention scores directly from compressed K without full decompression.
    """
    
    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.device = device
        
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        G = torch.randn(head_dim, head_dim, generator=gen)
        Q, R = torch.linalg.qr(G)
        diag_sign = torch.sign(torch.diag(R))
        diag_sign[diag_sign == 0] = 1.0
        self.Pi = (Q * diag_sign.unsqueeze(0)).to(device)
        
        self.centroids = self._solve_codebook(head_dim, self.mse_bits).to(device)
        
        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(seed + 10000)
        self.S = torch.randn(head_dim, head_dim, generator=gen2).to(device)
        
        self.PiT = self.Pi.T.contiguous()
    
    def _solve_codebook(self, d: int, bits: int) -> torch.Tensor:
        try:
            from scipy import integrate
            use_scipy = True
        except ImportError:
            use_scipy = False
        
        n_levels = 2 ** bits
        sigma = 1.0 / math.sqrt(d)
        
        if use_scipy:
            def pdf(x):
                return (1.0 / math.sqrt(2 * math.pi * sigma ** 2)) * math.exp(-x * x / (2 * sigma ** 2))
            
            lo, hi = -3.5 * sigma, 3.5 * sigma
            centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]
            
            for _ in range(200):
                boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
                edges = [lo * 3] + boundaries + [hi * 3]
                new_centroids = []
                for i in range(n_levels):
                    a, b = edges[i], edges[i + 1]
                    num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
                    den, _ = integrate.quad(pdf, a, b)
                    new_centroids.append(num / den if den > 1e-15 else centroids[i])
                
                if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < 1e-10:
                    break
                centroids = new_centroids
        else:
            lo, hi = -3.5 * sigma, 3.5 * sigma
            step = (hi - lo) / n_levels
            centroids = [lo + step * (i + 0.5) for i in range(n_levels)]
        
        return torch.tensor(centroids, dtype=torch.float32)
    
    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        """
        Compress states: (batch, heads, seq, head_dim) -> compressed dict.
        """
        B, H, S, D = states.shape
        flat = states.reshape(-1, D).float()
        
        vec_norms = torch.norm(flat, dim=-1, keepdim=True)
        flat_norm = flat / (vec_norms + 1e-8)
        
        rotated = flat_norm @ self.Pi.T
        diffs = rotated.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)
        
        reconstructed_rotated = self.centroids[indices.long()]
        k_mse = (reconstructed_rotated @ self.Pi) * vec_norms
        
        residual = flat - k_mse
        residual_norm = torch.norm(residual, dim=-1)
        
        projected = residual @ self.S.T
        signs = (projected >= 0).to(torch.int8) * 2 - 1
        
        return {
            "k_mse": k_mse.to(torch.float16).reshape(B, H, S, D),
            "qjl_signs": signs.reshape(B, H, S, D),
            "residual_norm": residual_norm.to(torch.float16).reshape(B, H, S),
            "shape": (B, H, S, D),
        }
    
    @torch.no_grad()
    def asymmetric_attention_scores(self, queries: torch.Tensor, compressed: dict) -> torch.Tensor:
        """
        Compute attention scores <Q, K> directly from compressed K.
        
        Uses the asymmetric estimator:
            <q, k> ≈ <q, k_mse> + ||r_k|| * sqrt(pi/2)/m * <S@q, signs_k>
        """
        k_mse = compressed["k_mse"].float()
        signs = compressed["qjl_signs"].float()
        r_norm = compressed["residual_norm"].float()
        
        term1 = torch.matmul(queries.float(), k_mse.transpose(-2, -1))
        
        q_projected = torch.matmul(queries.float(), self.S.T)
        qjl_ip = torch.matmul(q_projected, signs.transpose(-2, -1))
        
        m = self.S.shape[0]
        correction_scale = math.sqrt(math.pi / 2) / m
        term2 = correction_scale * qjl_ip * r_norm.unsqueeze(-2)
        
        return term1 + term2


class TurboQuantManager:
    """
    Unified Manager class for all cache compression.
    - TurboQuant KV cache compression
    - Image processing cache
    - Audio processing cache

    This manager serves as the single source of truth for all caching operations
    to avoid conflicts between different cache systems.
    """

    _instance: Optional['TurboQuantManager'] = None

    def __init__(self, enabled: bool = False, bits: int = 3, seed: int = 42):
        self.enabled = enabled
        self.bits = bits
        self.seed = seed
        self.caches: Dict[str, TurboQuantKVCache] = {}
        self.compressors_v2: Dict[str, TurboQuantCompressorV2] = {}
        self._image_cache: Dict[str, Any] = {}
        self._audio_cache: Dict[str, Any] = {}
        self._image_cache_limit = 50
        self._audio_cache_limit = 20
        self._memory_stats = {
            "original_bits": 0,
            "compressed_bits": 0,
            "compression_ratio": 1.0,
        }

    @classmethod
    def get_instance(cls) -> 'TurboQuantManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(enabled=False, bits=3, seed=42)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance."""
        if cls._instance is not None:
            cls._instance.clear_all()
        cls._instance = None
    
    def create_cache(self, name: str, d_key: int, d_value: int, device: str = "cpu") -> TurboQuantKVCache:
        """Create a new TurboQuant KV cache."""
        cache = TurboQuantKVCache(d_key, d_value, bits=self.bits, seed=self.seed, device=device)
        self.caches[name] = cache
        return cache
    
    def create_compressor_v2(self, name: str, head_dim: int, device: str = "cpu") -> TurboQuantCompressorV2:
        """Create a new V2 compressor for asymmetric attention."""
        compressor = TurboQuantCompressorV2(head_dim, bits=self.bits, seed=self.seed, device=device)
        self.compressors_v2[name] = compressor
        return compressor
    
    def get_cache(self, name: str) -> Optional[TurboQuantKVCache]:
        """Get cache by name."""
        return self.caches.get(name)
    
    def get_compressor_v2(self, name: str) -> Optional[TurboQuantCompressorV2]:
        """Get V2 compressor by name."""
        return self.compressors_v2.get(name)
    
    def clear_cache(self, name: str):
        """Clear a specific cache."""
        if name in self.caches:
            del self.caches[name]
        if name in self.compressors_v2:
            del self.compressors_v2[name]
    
    def clear_all(self):
        """Clear all caches including image and audio caches."""
        self.caches.clear()
        self.compressors_v2.clear()
        self._image_cache.clear()
        self._audio_cache.clear()

    def cache_image(self, key: str, value: Any):
        """Cache an image processing result."""
        if len(self._image_cache) >= self._image_cache_limit:
            oldest_key = next(iter(self._image_cache))
            del self._image_cache[oldest_key]
        self._image_cache[key] = value

    def get_cached_image(self, key: str) -> Optional[Any]:
        """Get a cached image result."""
        return self._image_cache.get(key)

    def cache_audio(self, key: str, value: Any):
        """Cache an audio processing result."""
        if len(self._audio_cache) >= self._audio_cache_limit:
            oldest_key = next(iter(self._audio_cache))
            del self._audio_cache[oldest_key]
        self._audio_cache[key] = value

    def get_cached_audio(self, key: str) -> Optional[Any]:
        """Get a cached audio result."""
        return self._audio_cache.get(key)
    
    def update_stats(self):
        """Update memory statistics."""
        total_original = 0
        total_compressed = 0
        
        for cache in self.caches.values():
            stats = cache.memory_usage_bits()
            total_original += stats["fp16_bits"]
            total_compressed += stats["total_bits"]
        
        if total_compressed > 0:
            self._memory_stats = {
                "original_bits": total_original,
                "compressed_bits": total_compressed,
                "compression_ratio": total_original / total_compressed,
            }
    
    def get_stats(self) -> dict:
        """Get current compression statistics."""
        self.update_stats()
        return self._memory_stats.copy()
    
    def enable(self):
        """Enable TurboQuant compression."""
        self.enabled = True
    
    def disable(self):
        """Disable TurboQuant compression."""
        self.enabled = False
    
    @property
    def is_enabled(self) -> bool:
        return self.enabled


_global_manager: Optional[TurboQuantManager] = None


def get_global_manager() -> TurboQuantManager:
    """Get or create the global TurboQuant manager instance (singleton)."""
    global _global_manager
    if _global_manager is None:
        _global_manager = TurboQuantManager(enabled=False, bits=3, seed=42)
    return _global_manager


def init_turboquant(enabled: bool = False, bits: int = 3, seed: int = 42) -> TurboQuantManager:
    """Initialize the global TurboQuant manager."""
    global _global_manager
    _global_manager = TurboQuantManager(enabled=enabled, bits=bits, seed=seed)
    return _global_manager


def clear_all_caches():
    """Clear all caches across the system."""
    manager = get_global_manager()
    manager.clear_all()
    try:
        from common import LLAMA_CPP_STORAGE
        LLAMA_CPP_STORAGE.clean_state()
    except Exception:
        pass