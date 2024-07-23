from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataReadConfig:
    train_dir: Path
    test_dir: Path
    index_separator: str
    image_id_index: int
    label_index: int

@dataclass(frozen=True)
class encodingsConfig:
    single_convolution: bool
    double_convolution: bool
    patch_convolution: bool
    pseudo_differential_privacy: bool
    pseudo_differential_privacy_patched: bool
    differential_privacy: bool
    differential_privacy_patched: bool
    differential_privacy_single_convolution: bool

@dataclass(frozen=True)
class saveEncodedImagesConfig:
    enabled: bool
    output_dir: Path





