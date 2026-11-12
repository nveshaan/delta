"""FAPrompt (Zhu et al., ICCV 2025), zero-shot inference with the authors' released prompt checkpoints.

Image-level score as in the authors' test.py: 0.5 * (text probability + weighted max patch similarity).
"""
import numpy as np
import torch
import torchvision.transforms as T

from baselines.common import choose_device, prepare_repo, select_checkpoint


class FAPrompt:
    def __init__(self, checkpoint, checkpoints, repo_url, repo_dir, device="auto", image_size=518,
                 features_list=(6, 12, 18, 24), depth=9, n_ctx=12, t_n_ctx=4, seed=111, **_):
        repo = prepare_repo(repo_url, repo_dir)
        import AnomalyCLIP_lib
        from AnomalyCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
        from AnomalyCLIP_lib.transform import image_transform
        from FAPrompt import FAPrompt as PromptLearner

        torch.manual_seed(seed)
        self.lib = AnomalyCLIP_lib
        self.device = choose_device(device)
        self.features_list = list(features_list)
        params = {"Prompt_length": n_ctx, "learnabel_text_embedding_depth": depth,
                  "learnabel_text_embedding_length": t_n_ctx}
        model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device="cpu", design_details=params,
                                        download_root=str(repo.parent / "clip_cache"))
        model.eval()
        learner = PromptLearner(model, params)
        file_id = select_checkpoint(checkpoint, checkpoints, "FAPrompt")
        path = repo / "checkpoints" / f"{checkpoint}.pth"
        if not path.is_file():
            import gdown
            path.parent.mkdir(exist_ok=True)
            gdown.download(id=file_id, output=str(path), quiet=False)
        learner.load_state_dict(torch.load(path, map_location="cpu")["prompt_learner"])
        model.float().to(self.device)
        learner.to(self.device)
        model.visual.DAPM_replace(DPAM_layer=20)
        self.model, self.learner = model, learner

        preprocess = image_transform(image_size, is_train=False, mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
        preprocess.transforms[0] = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)
        preprocess.transforms[1] = T.CenterCrop((image_size, image_size))
        self.preprocess = preprocess

    def _text_features(self, selected_tokens=None):
        pos, neg, tok_pos, tok_neg, compound, _ = (
            self.learner.forward() if selected_tokens is None else self.learner.forward(selected_tokens=selected_tokens)
        )
        text_pos = self.model.encode_text_learn(pos, tok_pos, compound).float()
        text_neg = self.model.encode_text_learn(neg, tok_neg, compound).float()
        return torch.cat([text_pos, torch.mean(text_neg, dim=0, keepdim=True)])

    @torch.no_grad()
    def _score_one(self, image) -> float:
        lib = self.lib
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        feats, patches = self.model.encode_image(x, self.features_list, DPAM_layer=20)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        text = self._text_features()
        text = torch.stack(torch.chunk(text, dim=0, chunks=2), dim=1)
        text = text / text.norm(dim=-1, keepdim=True)
        patches = patches / patches.norm(dim=-1, keepdim=True)
        similarity, _ = lib.compute_similarity_ori(patches, text[0])
        max_score1 = similarity[1:, :, 1].max(dim=0).values

        # Data-dependent abnormality prior: re-prompt with the 10 most abnormal patch tokens of each layer.
        _, top_idx = torch.topk(similarity[1:, :, 1], k=10, dim=0, largest=True, sorted=True)
        tokens = patches[1:, :, :].permute(1, 0, 2)
        selected = torch.stack([tokens[i][top_idx.permute(1, 0)[i]] for i in range(tokens.shape[0])])
        text = torch.stack([self._text_features(selected_tokens=s) for s in selected])
        text = text / text.norm(dim=-1, keepdim=True)

        probs = (feats @ text.permute(0, 2, 1) / 0.07).softmax(-1)[:, 0, 1]
        similarity, _ = lib.compute_similarity(patches, text)
        max_score2 = similarity[1:, :, 1].max(dim=0).values
        max_score = (2 * max_score1 + max_score2) / 3
        return float((0.5 * (probs + max_score)).mean())

    def score(self, images) -> np.ndarray:
        return np.asarray([self._score_one(image) for image in images])
