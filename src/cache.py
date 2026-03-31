# src/cache.py
# LRU cache keyed by image SHA256 hash
# Prevents recomputing WideResNet + CLIP for repeated images
# maxsize=128: holds ~128 inference results in RAM (~100MB max)

import hashlib
from collections import OrderedDict
from PIL import Image
import io


MAX_CACHE_SIZE = 128


class LRUCache:
    """
    Simple LRU cache backed by OrderedDict.
    Key: SHA256 hash of raw image bytes
    Value: dict of precomputed features for that image
    
    Why not functools.lru_cache: we need explicit key control
    (image hash, not the PIL object itself which is unhashable).
    """

    def __init__(self, maxsize=MAX_CACHE_SIZE):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key not in self.cache:
            self.misses += 1
            return None
        # Move to end = most recently used
        self.cache.move_to_end(key)
        self.hits += 1
        return self.cache[key]

    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            # Pop least recently used (first item)
            self.cache.popitem(last=False)

    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": round(hit_rate, 4),
            "current_size": len(self.cache),
            "max_size": self.maxsize
        }

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0


def get_image_hash(image_bytes: bytes) -> str:
    """
    SHA256 hash of raw image bytes.
    Used as cache key AND as unique image ID in HF Dataset logs.
    Same image submitted twice = same hash = cache hit.
    """
    return hashlib.sha256(image_bytes).hexdigest()


def pil_to_bytes(pil_img: Image.Image) -> bytes:
    """Convert PIL image to bytes for hashing."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# Global cache instance — lives for the entire FastAPI server lifetime
# Initialised once in api/startup.py, imported everywhere
inference_cache = LRUCache(maxsize=MAX_CACHE_SIZE)