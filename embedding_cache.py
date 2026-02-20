"""
EmbeddingCache: Cache video/audio embeddings to disk.

Cache path format: {base_dir}/{video_id}/{modality}/{start}_{end}.pt
Example: /mnt/hdd4tb/jaemo/data/LLP/feats/video001/audio/4_5.pt
"""

import torch
import os
from typing import Callable, Optional


class EmbeddingCache:
    def __init__(self, backbone: str, cache_dir: str = "/mnt/hdd4tb/jaemo/data/LLP/cached_avvp", enabled: bool = True):
        """
        Args:
            backbone: Model backbone name ('language_bind' or 'clip_clap')
            cache_dir: Base directory for caching
            enabled: Whether caching is enabled
        """
        self.backbone = backbone
        self.cache_dir = cache_dir
        self.enabled = enabled
        
    def _make_path(self, video_id: str, modality: str, start: float, end: float) -> str:
        """
        Create cache file path.
        Format: {cache_dir}/{modality}/{backbone}/{video_id}/{start}_{end}.pt
        
        Example: /mnt/hdd4tb/jaemo/data/LLP/feats/audio/clap/video001/4_5.pt
        
        Args:
            video_id: Unique identifier for the video
            modality: 'video' or 'audio'
            start: Start time in seconds
            end: End time in seconds
        """
        # Convert to int if they are whole numbers for cleaner filenames
        start_str = str(int(start)) if start == int(start) else f"{start:.2f}"
        end_str = str(int(end)) if end == int(end) else f"{end:.2f}"
        
        return os.path.join(
            self.cache_dir,
            modality,
            self.backbone,
            video_id,
            f"{start_str}_{end_str}.pt"
        )
    
    def get(self, video_id: str, modality: str, start: float, end: float) -> Optional[torch.Tensor]:
        """
        Get cached embedding from disk if exists.
        
        Returns:
            Cached tensor if exists, None otherwise
        """
        if not self.enabled:
            return None
            
        cache_path = self._make_path(video_id, modality, start, end)
        
        if os.path.exists(cache_path):
            try:
                return torch.load(cache_path, weights_only=False)
            except Exception as e:
                print(f"Cache load failed: {cache_path}, {e}")
                return None
        return None
    
    def put(self, video_id: str, modality: str, start: float, end: float, embedding: torch.Tensor) -> None:
        """
        Store embedding to disk.
        """
        if not self.enabled:
            return
            
        cache_path = self._make_path(video_id, modality, start, end)
        
        try:
            # Create directory if not exists
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            # Save tensor (CPU to save space)
            torch.save(embedding.detach().cpu(), cache_path)
            # print(f"[Cache] Saved: {cache_path}")
        except Exception as e:
            print(f"Cache save failed: {cache_path}, {e}")
    
    def get_or_compute(
        self, 
        video_id: str, 
        modality: str, 
        compute_fn: Callable[[], torch.Tensor],
        start: float,
        end: float
    ) -> torch.Tensor:
        """
        Get cached embedding or compute and cache it.
        
        Returns:
            Embedding tensor (on CPU, caller should move to device)
        """
        # Try to get from cache
        cached = self.get(video_id, modality, start, end)
        if cached is not None:
            return cached
        
        # Compute embedding
        embedding = compute_fn()
        
        # Store in cache
        self.put(video_id, modality, start, end, embedding)
        
        return embedding
    
    def __repr__(self) -> str:
        return f"EmbeddingCache(backbone={self.backbone}, cache_dir={self.cache_dir}, enabled={self.enabled})"
