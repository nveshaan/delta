#!/usr/bin/env python3
"""Generate medical-image embeddings from raw-data configs.

Each configured parent dataset is embedded as one collection. Fine-grained
labels from nested YAML, CSV, and MedIAnomaly metadata are stored alongside
each embedding matrix. Runtime, path, and model values are in
configs/embeddings.yaml.

Outputs:
    data/<modality>/<dataset_name>/<encoder>_embeds.npy
    data/<modality>/<dataset_name>/labels.npy
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "embeddings.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Encoder:
    name: str
    model: Any
    preprocess: Callable
    encode_batch: Callable[[torch.Tensor], Any]


def load_settings(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Embedding config does not exist: {path}")
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_.") or "dataset"


def check_optional_dependencies() -> None:
    try:
        import einops  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "MedImageInsight requires the `einops` package. "
            "Install it with: pip install einops"
        ) from exc


def is_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return (
        "out of memory" in message
        or ("mps" in message and "memory" in message)
        or "cuda out of memory" in message
    )


def unload_encoder(encoder: Encoder | None, device: torch.device | None = None) -> None:
    """Release model references and clear accelerator allocator state."""
    if encoder is None:
        return
    try:
        model = encoder.model
        try:
            model.to("cpu")
        except Exception:
            pass
        del model
    except Exception:
        pass
    try:
        del encoder
    except Exception:
        pass
    gc.collect()

    if device is not None and device.type == "mps":
        if hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
            try:
                torch.mps.synchronize()
            except Exception:
                pass
    elif device is not None and device.type == "cuda":
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass


# =============================================================================
# MULTIMODAL ENCODER LOADERS
# Self-contained implementations copied from plot_multimodal_embeddings_before_after_msde.py.
# All hardcoded values, repo targets, and models are loaded from configs/embeddings.yaml.
# =============================================================================

def load_medimageinsight(device: torch.device, config: dict[str, Any]) -> Encoder:
    """Load local MedImageInsight from Hugging Face snapshot."""
    check_optional_dependencies()
    from huggingface_hub import snapshot_download

    repo_dir = Path(snapshot_download(repo_id=config["hf_repo"]))
    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    from medimageinsightmodel import MedImageInsight

    model_dir = repo_dir / config.get("model_subdirectory", "2024.09.27")
    classifier = MedImageInsight(
        model_dir=str(model_dir),
        vision_model_name=config["vision_model_name"],
        language_model_name=config["language_model_name"],
    )
    classifier.load_model()
    classifier.device = device
    classifier.model = classifier.model.to(device)
    classifier.model.eval()

    def encode_batch(images: torch.Tensor) -> torch.Tensor:
        return classifier.model.encode_image(images.to(device))

    return Encoder("MedImageInsight", classifier.model, classifier.preprocess, encode_batch)


def load_medsiglip(device: torch.device, config: dict[str, Any]) -> Encoder:
    """Load MedSigLIP vision encoder using Hugging Face Transformers."""
    from transformers import AutoModel, AutoProcessor

    token_env = config.get("token_env", "HF_TOKEN")
    token = config.get("token") or (os.getenv(token_env) if token_env else None)
    kwargs = {"token": token} if token else {}

    processor = AutoProcessor.from_pretrained(config["model_name"], **kwargs)
    try:
        model = AutoModel.from_pretrained(config["model_name"], **kwargs).to(device)
    except OSError as exc:
        raise RuntimeError(
            "MedSigLIP is a gated Hugging Face model. "
            "Accept the Health AI Developer Foundations terms for "
            f"{config['model_name']} and authenticate with either "
            "`huggingface-cli login` or HF_TOKEN, then rerun."
        ) from exc
    model.eval()

    def preprocess(image):
        return processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    def encode_batch(images: torch.Tensor) -> torch.Tensor:
        outputs = model.get_image_features(pixel_values=images.to(device))
        if hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        elif hasattr(outputs, "image_embeds"):
            features = outputs.image_embeds
        elif isinstance(outputs, tuple):
            features = outputs[0]
        else:
            features = outputs
        if features is None or not torch.is_tensor(features):
            raise RuntimeError(
                f"MedSigLIP did not return a tensor image embedding. Got: {type(outputs).__name__}"
            )
        return features

    return Encoder("MedSigLIP", model, preprocess, encode_batch)


def _purge_open_clip_modules(unimed_repo_dir: Path) -> None:
    unimed_src = str((unimed_repo_dir / "src").resolve())
    sys.path[:] = [entry for entry in sys.path if str(Path(entry).resolve()) != unimed_src]
    for name in list(sys.modules):
        if name == "open_clip" or name.startswith("open_clip."):
            del sys.modules[name]


def _import_official_open_clip(unimed_repo_dir: Path):
    _purge_open_clip_modules(unimed_repo_dir)
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "The official pip `open_clip_torch` package is required for BiomedCLIP. "
            "Install it with: pip install -U open_clip_torch"
        ) from exc
    module_file = getattr(open_clip, "__file__", "") or ""
    unimed_src = str((unimed_repo_dir / "src").resolve())
    try:
        module_path = str(Path(module_file).resolve())
    except Exception:
        module_path = module_file
    if module_path.startswith(unimed_src):
        raise RuntimeError(
            "UniMed-CLIP's bundled OpenCLIP fork is shadowing the official pip OpenCLIP package. "
            "Remove the UniMed src path from PYTHONPATH and rerun."
        )
    return open_clip


def load_biomedclip(
    device: torch.device, config: dict[str, Any], unimed_repo_dir: Path
) -> Encoder:
    """Load BiomedCLIP using the official open_clip package."""
    open_clip = _import_official_open_clip(unimed_repo_dir)
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            config["model_name"], device=device
        )
    except Exception as exc:
        raise RuntimeError(
            f"BiomedCLIP could not be loaded from '{config['model_name']}': {type(exc).__name__}: {exc}"
        ) from exc
    model = model.to(device)
    model.eval()

    def encode_batch(images: torch.Tensor) -> torch.Tensor:
        return model.encode_image(images.to(device))

    return Encoder("BiomedCLIP", model, preprocess, encode_batch)


def ensure_unimed_repo(config: dict[str, Any]) -> Path:
    repo_dir = project_path(config["repository_dir"])
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", config["git_repository"], str(repo_dir)],
            check=True,
        )
    return repo_dir


def _create_unimed_model_with_legacy_checkpoint(
    create_model_and_transforms,
    model_name: str,
    weights_path: str,
    *,
    precision: str,
    device: torch.device,
    force_quick_gelu: bool,
    pretrained_image: bool,
    mean: Any,
    std: Any,
    inmem: bool,
    text_encoder_name: str,
    legacy_checkpoints: tuple[str, ...],
):
    original_torch_load = torch.load

    def trusted_load(*args, **kwargs):
        checkpoint = args[0] if args else kwargs.get("f")
        checkpoint_str = str(checkpoint)
        if any(checkpoint_str.endswith(name) for name in legacy_checkpoints):
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = trusted_load
    try:
        return create_model_and_transforms(
            model_name,
            weights_path,
            precision=precision,
            device=device,
            force_quick_gelu=force_quick_gelu,
            pretrained_image=pretrained_image,
            mean=mean,
            std=std,
            inmem=inmem,
            text_encoder_name=text_encoder_name,
        )
    finally:
        torch.load = original_torch_load


def load_unimedclip(device: torch.device, config: dict[str, Any]) -> Encoder:
    """Load UniMed-CLIP using its bundled OpenCLIP fork and legacy weights handling."""
    from huggingface_hub import hf_hub_download

    repo_dir = ensure_unimed_repo(config)
    src_dir = repo_dir / "src"
    if not src_dir.is_dir():
        raise FileNotFoundError(f"UniMed-CLIP source directory missing: {src_dir}")
    sys.path.insert(0, str(src_dir))
    for name in list(sys.modules):
        if name == "open_clip" or name.startswith("open_clip."):
            del sys.modules[name]
    from open_clip import create_model_and_transforms, get_mean_std

    weights_path = hf_hub_download(
        repo_id=config["hf_repo"], filename=config["hf_filename"]
    )
    mean, std = get_mean_std()
    legacy_checkpoints = tuple(
        config.get(
            "legacy_checkpoints",
            (
                "unimed-clip-vit-b16.pt",
                "unimed_clip_vit_l14_base_text_encoder.pt",
                "unimed_clip_vit_l14_large_text_encoder.pt",
            ),
        )
    )

    model, _, preprocess = _create_unimed_model_with_legacy_checkpoint(
        create_model_and_transforms,
        config["model_name"],
        weights_path,
        precision=config.get("precision", "amp"),
        device=device,
        force_quick_gelu=config.get("force_quick_gelu", True),
        pretrained_image=config.get("pretrained_image", False),
        mean=mean,
        std=std,
        inmem=config.get("in_memory", True),
        text_encoder_name=config["text_encoder_name"],
        legacy_checkpoints=legacy_checkpoints,
    )
    model.eval()

    def encode_batch(images: torch.Tensor) -> torch.Tensor:
        return model.encode_image(images.to(device))

    return Encoder("UniMedCLIP", model, preprocess, encode_batch)


def load_clip(device: torch.device, config: dict[str, Any]) -> Encoder:
    """Load standard CLIP model."""
    try:
        import clip
    except ImportError as exc:
        raise RuntimeError("CLIP requires the `openai-clip` package.") from exc
    model, preprocess = clip.load(config["model_name"], device=device)
    model.eval()

    def encode_batch(images: torch.Tensor) -> torch.Tensor:
        return model.encode_image(images.to(device))

    return Encoder("CLIP", model, preprocess, encode_batch)


def load_encoder(name: str, device: torch.device, settings: dict[str, Any]) -> Encoder:
    encoders_cfg = settings["encoders"]
    if name not in encoders_cfg:
        raise KeyError(f"Unknown encoder: {name}. Available: {list(encoders_cfg.keys())}")
    cfg = encoders_cfg[name]

    if name == "MedImageInsight":
        return load_medimageinsight(device, cfg)
    if name == "MedSigLIP":
        return load_medsiglip(device, cfg)
    if name == "BiomedCLIP":
        unimed_repo = project_path(encoders_cfg["UniMedCLIP"]["repository_dir"])
        return load_biomedclip(device, cfg, unimed_repo)
    if name == "UniMedCLIP":
        return load_unimedclip(device, cfg)
    if name == "CLIP":
        return load_clip(device, cfg)
    raise KeyError(f"Unsupported encoder loader for: {name}")


# =============================================================================
# BATCH ENCODING & ADAPTIVE OOM RECOVERY
# =============================================================================

def output_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    for attribute in ("pooler_output", "image_embeds", "last_hidden_state"):
        candidate = getattr(value, attribute, None)
        if torch.is_tensor(candidate):
            return candidate
    if isinstance(value, (tuple, list)):
        return next(item for item in value if torch.is_tensor(item))
    if isinstance(value, dict):
        return next(item for item in value.values() if torch.is_tensor(item))
    raise TypeError(f"Encoder returned unsupported output: {type(value).__name__}")


def encode_dataset(
    dataset,
    encoder: Encoder,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    normalize: bool = True,
    norm_eps: float = 1e-12,
    output_dtype: str = "float32",
) -> np.ndarray:
    """Encode all images in dataset with adaptive batch halving on OOM.

    Retains completed batches on CPU if memory pressure forces a smaller
    batch size for subsequent items.
    """
    total = len(dataset)
    if total == 0:
        return np.empty((0, 0), dtype=getattr(np, output_dtype, np.float32))

    current_batch_size = max(1, batch_size)
    start = 0
    chunks: list[torch.Tensor] = []

    while start < total:
        batch_end = min(start + current_batch_size, total)
        batch_indices = list(range(start, batch_end))

        try:
            subset = Subset(dataset, batch_indices)
            loader = DataLoader(
                subset,
                batch_size=len(batch_indices),
                shuffle=False,
                num_workers=num_workers,
            )
            with torch.inference_mode():
                for images, _ in loader:
                    features = output_tensor(encoder.encode_batch(images))
                    if features.ndim > 2:
                        features = features.flatten(1)
                    chunks.append(features.detach().float().cpu())

            start = batch_end

        except RuntimeError as error:
            if not is_oom(error) or current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            print(f"      reducing batch size to {current_batch_size} after OOM")
            unload_encoder(None, device)

    all_features = torch.cat(chunks, dim=0)

    if not torch.isfinite(all_features).all():
        raise ValueError(f"Non-finite embeddings produced by {encoder.name}")

    if normalize:
        all_features = F.normalize(all_features, p=2, dim=-1, eps=norm_eps)

    target_np_dtype = getattr(np, output_dtype, np.float32)
    return all_features.numpy().astype(target_np_dtype, copy=False)


# =============================================================================
# DATASET ACCESS & PIPELINE EXECUTION
# =============================================================================

def dataset_groups(modality: str, settings: dict[str, Any]):
    from datasets.raw_common import RawDataset

    config_dir = project_path(settings["dataset_config_dir"])
    config_pattern = settings.get("dataset_config_pattern", "raw_{modality}.yaml")
    config_file = config_dir / config_pattern.format(modality=modality)
    if not config_file.is_file():
        raise FileNotFoundError(f"Modality config does not exist: {config_file}")

    config = OmegaConf.load(config_file)
    for dataset_name, subtypes in config.sources.items():
        yield dataset_name, subtypes, RawDataset


def generate_for_encoder(
    name: str,
    modalities: list[str],
    dataset_filter: str | None,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    overwrite: bool,
    settings: dict[str, Any],
) -> None:
    print(f"Loading {name} on {device}...")
    encoder = load_encoder(name, device, settings)
    output_root = project_path(settings["output_root"])
    defaults = settings.get("defaults", {})
    normalize = defaults.get("normalize", True)
    norm_eps = float(defaults.get("norm_eps", 1e-12))
    output_dtype = defaults.get("output_dtype", "float32")
    labels_dtype = getattr(np, defaults.get("labels_dtype", "int64"), np.int64)

    try:
        for modality in modalities:
            for dataset_name, subtypes, RawDataset in dataset_groups(modality, settings):
                if dataset_filter:
                    requested = dataset_filter.strip().lower()
                    if requested not in {dataset_name.lower(), safe_name(dataset_name).lower()}:
                        continue

                output_dir = output_root / modality / dataset_name
                output_dir.mkdir(parents=True, exist_ok=True)

                embeddings_path = output_dir / f"{safe_name(encoder.name)}_embeds.npy"
                labels_path = output_dir / "labels.npy"

                if embeddings_path.exists() and labels_path.exists() and not overwrite:
                    print(f"  {modality}/{dataset_name}: cached")
                    continue

                dataset = RawDataset(subtypes, transform=encoder.preprocess)
                labels = np.asarray(dataset.labels, dtype=labels_dtype)
                print(f"  {modality}/{dataset_name}: {len(dataset)} images")

                embeddings = encode_dataset(
                    dataset=dataset,
                    encoder=encoder,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    device=device,
                    normalize=normalize,
                    norm_eps=norm_eps,
                    output_dtype=output_dtype,
                )

                if len(embeddings) != len(labels):
                    raise RuntimeError(
                        f"Embedding/label mismatch for {modality}/{dataset_name}: "
                        f"{len(embeddings)} embeds vs {len(labels)} labels"
                    )

                if labels_path.exists():
                    existing_labels = np.load(labels_path)
                    if not np.array_equal(existing_labels, labels):
                        raise RuntimeError(
                            f"Label order mismatch for {modality}/{dataset_name}; "
                            "dataset samples are not ordered consistently"
                        )
                else:
                    np.save(labels_path, labels)
                np.save(embeddings_path, embeddings)
                print(f"    saved {embeddings_path} and {labels_path}")
    finally:
        unload_encoder(encoder, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--modality", action="append")
    parser.add_argument("--encoder", action="append")
    parser.add_argument("--dataset", help="Parent folder name or sanitized filename stem")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    defaults = settings.get("defaults", {})

    seed = args.seed if args.seed is not None else int(defaults.get("seed", 42))
    seed_everything(seed)

    modalities = args.modality or list(settings["modalities"])
    encoders = args.encoder or list(settings["encoders"])

    unknown_modalities = set(modalities) - set(settings["modalities"])
    unknown_encoders = set(encoders) - set(settings["encoders"])
    if unknown_modalities:
        raise ValueError(f"Unknown modalities: {sorted(unknown_modalities)}")
    if unknown_encoders:
        raise ValueError(f"Unknown encoders: {sorted(unknown_encoders)}")

    batch_size = args.batch_size or int(defaults.get("batch_size", 16))
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")

    num_workers = int(defaults.get("num_workers", 0))
    device = choose_device(args.device or defaults.get("device", "auto"))

    for encoder_name in encoders:
        generate_for_encoder(
            name=encoder_name,
            modalities=modalities,
            dataset_filter=args.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            overwrite=args.overwrite,
            settings=settings,
        )


if __name__ == "__main__":
    main()
