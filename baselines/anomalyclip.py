"""AnomalyCLIP (Zhou et al., ICLR 2024), zero-shot inference with the authors' released prompt checkpoints.

Image-level score as in the authors' test.py: softmax probability of the learned "abnormal" text prompt.
"""
import numpy as np
import torch
import torchvision.transforms as T

from baselines.common import choose_device, prepare_repo, select_checkpoint


class AnomalyCLIP:
    def __init__(self, checkpoint, checkpoints, repo_url, repo_dir, device="auto", image_size=518,
                 features_list=(6, 12, 18, 24), depth=9, n_ctx=12, t_n_ctx=4, batch_size=16, seed=111, **_):
        repo = prepare_repo(repo_url, repo_dir)
        import AnomalyCLIP_lib
        from AnomalyCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
        from AnomalyCLIP_lib.transform import image_transform
        from prompt_ensemble import AnomalyCLIP_PromptLearner

        torch.manual_seed(seed)
        self.device = choose_device(device)
        self.batch_size = batch_size
        self.features_list = list(features_list)
        params = {"Prompt_length": n_ctx, "learnabel_text_embedding_depth": depth,
                  "learnabel_text_embedding_length": t_n_ctx}
        model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device="cpu", design_details=params,
                                        download_root=str(repo.parent / "clip_cache"))
        model.eval()
        prompt_learner = AnomalyCLIP_PromptLearner(model, params)
        state = torch.load(repo / select_checkpoint(checkpoint, checkpoints, "AnomalyCLIP"), map_location="cpu")
        prompt_learner.load_state_dict(state["prompt_learner"])
        model.float().to(self.device)
        prompt_learner.to(self.device)
        model.visual.DAPM_replace(DPAM_layer=20)
        self.model = model

        with torch.no_grad():
            prompts, tokenized, compound = prompt_learner(cls_id=None)
            text = model.encode_text_learn(prompts, tokenized, compound).float()
            text = torch.stack(torch.chunk(text, dim=0, chunks=2), dim=1)[0]  # (2, D): normal, abnormal
            self.text = text / text.norm(dim=-1, keepdim=True)

        preprocess = image_transform(image_size, is_train=False, mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
        preprocess.transforms[0] = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)
        preprocess.transforms[1] = T.CenterCrop((image_size, image_size))
        self.preprocess = preprocess

    @torch.no_grad()
    def score(self, images) -> np.ndarray:
        scores = []
        for i in range(0, len(images), self.batch_size):
            x = torch.stack([self.preprocess(img) for img in images[i:i + self.batch_size]]).to(self.device)
            feats, _ = self.model.encode_image(x, self.features_list, DPAM_layer=20)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            probs = (feats.float() @ self.text.T / 0.07).softmax(-1)[:, 1]
            scores.append(probs.cpu().numpy())
        return np.concatenate(scores)
