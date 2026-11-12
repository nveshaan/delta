"""AdaCLIP (Cao et al., ECCV 2024), zero-shot inference with the authors' released weights.

Image-level score is the model's aggregated anomaly score for the image, prompted with `class_name`.
"""
import json

import numpy as np
import torch

from baselines.common import choose_device, prepare_repo, select_checkpoint


class AdaCLIP:
    def __init__(self, checkpoint, checkpoints, repo_url, repo_dir, weights_repo, class_name="object",
                 device="auto", image_size=518, backbone="ViT-L-14-336", prompting_depth=4, prompting_length=5,
                 prompting_type="SD", prompting_branch="VL", use_hsf=True, k_clusters=20, batch_size=16, **_):
        repo = prepare_repo(repo_url, repo_dir)
        from huggingface_hub import hf_hub_download
        from method import AdaCLIP_Trainer

        self.device = choose_device(device)
        self.class_name = class_name
        self.batch_size = batch_size
        with open(repo / "model_configs" / f"{backbone}.json") as f:
            cfg = json.load(f)
        layers = cfg["vision_cfg"]["layers"] // 4
        self.model = AdaCLIP_Trainer(
            backbone=backbone, feat_list=[layers, layers * 2, layers * 3, layers * 4],
            input_dim=cfg["vision_cfg"]["width"], output_dim=cfg["embed_dim"], learning_rate=0.0,
            device=self.device, image_size=image_size, prompting_depth=prompting_depth,
            prompting_length=prompting_length, prompting_branch=prompting_branch, prompting_type=prompting_type,
            use_hsf=use_hsf, k_clusters=k_clusters,
        ).to(self.device)
        weights = hf_hub_download(weights_repo, select_checkpoint(checkpoint, checkpoints, "AdaCLIP"),
                                  repo_type="space", local_dir=repo.parent / "adaclip_weights")
        self.model.load(weights)
        self.model.eval()

    @torch.no_grad()
    def score(self, images) -> np.ndarray:
        scores = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            x = torch.stack([self.model.preprocess(img) for img in batch]).to(self.device)
            _, anomaly_score = self.model.clip_model(x, [self.class_name] * len(batch), aggregation=True)
            scores.append(anomaly_score.float().cpu().numpy())
        return np.concatenate(scores)
