# -*- coding: utf-8 -*-
"""
Stage-3 Joint Optimization Script: ViPER Scalar (\alpha) & CIDEr Reinforcement Learning (SCST).
Follows the "align first, decode second, perceive last" progressive optimization path
described in Sections 3.2, 4.2, and 4.7 of the paper:
  - Loads the converged Stage-1 Beta-SEA adapter checkpoint.
  - Initializes the ViPER scaling factor \sigma(\alpha) at the warm-start value of 0.02
    via inverse Sigmoid projection with upper bound \lambda_max = 0.5 (Eq. 3).
  - Applies differential learning rates: 1e-4 for the ViPER scalar and 2e-4 for Beta-SEA.
  - Supports both Self-Critical Sequence Training (SCST) for direct CIDEr + Visual Critic
    reward optimization (Table 2 & Fig. 10) and standard Cross-Entropy (CE) joint tuning (Table 3).
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from pycocoevalcap.cider.cider import Cider

try:
    from models.blip import blip_decoder
except ImportError as exc:
    raise ImportError(
        "Cannot import 'models.blip'. Please run this script from the project root directory."
    ) from exc

from core.modules import ViPER, BetaSemanticAdapter
from core.evaluator import strip_blip_prompt
from data_pipeline.syntactic_dataset import build_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Stage3-ViPER-SCST")


# ==============================================================================
# 1. Dataset & Collation for Stage-3 Joint Optimization (CE & SCST)
# ==============================================================================
class Stage3COCODataset(Dataset):
    """
    MS COCO dataset loader for Stage-3 joint optimization.
    Provides each image alongside its full set of 5 human reference captions
    (required for CIDEr reward computation in SCST) and one primary caption for CE.
    """

    def __init__(self, img_root: str, ann_json_path: str, max_samples: int = 0) -> None:
        super().__init__()
        self.img_root = img_root
        if not os.path.exists(ann_json_path):
            raise FileNotFoundError(f"Annotation file not found: {ann_json_path}")

        with open(ann_json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        img_map: Dict[Any, Dict[str, Any]] = {}
        if isinstance(raw_data, list):
            for item in raw_data:
                img_key = item.get("image_id", item.get("image") or item.get("file_name"))
                file_name = item.get("image") or item.get("file_name")
                caption = item["caption"].strip()
                if img_key not in img_map:
                    img_map[img_key] = {"image_id": img_key, "file_name": file_name, "refs": []}
                img_map[img_key]["refs"].append(caption)
        elif isinstance(raw_data, dict) and "annotations" in raw_data:
            id_to_file = {img["id"]: img["file_name"] for img in raw_data.get("images", [])}
            for ann in raw_data["annotations"]:
                img_id = ann["image_id"]
                if img_id not in img_map:
                    img_map[img_id] = {
                        "image_id": img_id,
                        "file_name": id_to_file.get(img_id, ""),
                        "refs": [],
                    }
                img_map[img_id]["refs"].append(ann["caption"].strip())
        else:
            raise ValueError("Unsupported annotation JSON format.")

        self.samples = list(img_map.values())
        if max_samples > 0:
            self.samples = self.samples[:max_samples]
        logger.info("Loaded %d unique images with reference caption sets.", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        img_path = os.path.join(self.img_root, sample["file_name"])
        return {
            "image_id": str(sample["image_id"]),
            "img_path": img_path,
            "primary_caption": sample["refs"][0],
            "all_references": sample["refs"],
        }


class Stage3Collator:
    """Collates image tensors, primary captions, and reference lists for SCST."""

    def __init__(self, transform: Any) -> None:
        self.transform = transform

    def __call__(self, batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        images: List[torch.Tensor] = []
        image_ids: List[str] = []
        primary_captions: List[str] = []
        references_dict: Dict[str, List[str]] = {}

        for idx, item in enumerate(batch):
            try:
                raw_img = Image.open(item["img_path"]).convert("RGB")
                images.append(self.transform(raw_img))
                uid = f"{item['image_id']}_{idx}"
                image_ids.append(uid)
                primary_captions.append(item["primary_caption"])
                references_dict[uid] = item["all_references"]
            except Exception as exc:
                logger.warning("Skipping unreadable image '%s': %s", item.get("img_path"), exc)
                continue

        if not images:
            return None

        return {
            "images": torch.stack(images, dim=0),
            "image_ids": image_ids,
            "primary_captions": primary_captions,
            "references_dict": references_dict,
        }


# ==============================================================================
# 2. Joint ViPER + Beta-SEA + Gamma-DHR Closed-Loop Model Wrapper
# ==============================================================================
class BetaBLIPStage3Model(nn.Module):
    """
    End-to-end Stage-3 model integrating:
      - Frozen ViT-B/16 visual encoder + frozen BERT-Base text decoder.
      - Trainable ViPER module (single scalar \alpha with inverse-Sigmoid warm start at 0.02).
      - Trainable Beta-SEA cross-attention adapter (5M parameters initialized from Stage 1).
    """

    def __init__(
        self,
        base_model_path: str,
        stage1_adapter_ckpt: str,
        image_size: int = 384,
        vit_type: str = "base",
        warm_start_scale: float = 0.02,
        lambda_max: float = 0.5,
        equilibrium_target: float = 0.083,
    ) -> None:
        super().__init__()
        logger.info("Loading frozen BLIP backbone from: %s", base_model_path)
        self.blip = blip_decoder(pretrained=base_model_path, image_size=image_size, vit=vit_type)

        # Initialize trainable ViPER module at warm-start value 0.02 (Eq. 1-3 & Section 4.2)
        self.viper = ViPER(
            scale=warm_start_scale,
            lambda_max=lambda_max,
            trainable=True,
            use_variance_routing=True,
        )
        self.equilibrium_target = equilibrium_target

        # Initialize Beta-SEA adapter and load Stage-1 pre-aligned weights
        self.adapter = BetaSemanticAdapter(embed_dim=768)
        if os.path.exists(stage1_adapter_ckpt):
            logger.info("Loading Stage-1 converged Beta-SEA weights: %s", stage1_adapter_ckpt)
            ckpt = torch.load(stage1_adapter_ckpt, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.adapter.load_state_dict(state_dict, strict=False)
        else:
            logger.warning(
                "Stage-1 checkpoint '%s' not found; starting with freshly initialized Beta-SEA.",
                stage1_adapter_ckpt,
            )

        # Freeze entire 224M BLIP backbone (ViT + BERT weights)
        for param in self.blip.parameters():
            param.requires_grad = False

        # Enable gradients ONLY for ViPER scalar \alpha and Beta-SEA adapter
        for param in self.viper.parameters():
            param.requires_grad = True
        for param in self.adapter.parameters():
            param.requires_grad = True

    def encode_and_inject_prior(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes frozen ViT extraction -> trainable ViPER calibration (Eq. 3) ->
        trainable Beta-SEA feed-forward task prior injection (Eq. 9).

        Returns:
            image_embeds_calibrated: ViPER-calibrated patch features \tilde{V} [B, N, D].
            image_embeds_enhanced: Prior-injected visual context for decoding [B, N, D].
        """
        with torch.no_grad():
            image_embeds_raw = self.blip.visual_encoder(images)

        # Differentiable ViPER modulation (gradients flow to self.viper.raw_alpha)
        image_embeds_calibrated = self.viper(image_embeds_raw)

        # Feed-forward task prompt prior injection (Eq. 9 in Section 3.4)
        batch_size = images.size(0)
        prompts = [self.blip.prompt] * batch_size
        prompt_inputs = self.blip.tokenizer(prompts, return_tensors="pt", padding=True).to(
            images.device
        )

        with torch.no_grad():
            prompt_out = self.blip.text_decoder.bert(
                input_ids=prompt_inputs.input_ids,
                attention_mask=prompt_inputs.attention_mask,
                return_dict=True,
                encoder_hidden_states=None,
            )
            prompt_queries = prompt_out.last_hidden_state[:, 0, :].unsqueeze(1)

        prior_residual = self.adapter(image_embeds_calibrated, prompt_queries).squeeze(1)

        # Differentiable residual fusion onto the global [CLS] token
        cls_enhanced = image_embeds_calibrated[:, :1, :] + prior_residual.unsqueeze(1)
        image_embeds_enhanced = torch.cat([cls_enhanced, image_embeds_calibrated[:, 1:, :]], dim=1)

        return image_embeds_calibrated, image_embeds_enhanced

    def compute_sequence_log_probs(
        self, image_embeds_enhanced: torch.Tensor, captions: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes differentiable sequence log-probabilities and Cross-Entropy loss
        conditioned on `image_embeds_enhanced`, allowing gradients to flow back
        through the frozen BERT decoder's cross-attention into Beta-SEA and ViPER.
        """
        device = image_embeds_enhanced.device
        prompt_prefix = self.blip.prompt
        full_texts = [prompt_prefix + cap for cap in captions]

        text_inputs = self.blip.tokenizer(
            full_texts,
            padding="longest",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(device)

        input_ids = text_inputs.input_ids.clone()
        input_ids[:, 0] = self.blip.tokenizer.bos_token_id

        decoder_targets = input_ids.masked_fill(
            input_ids == self.blip.tokenizer.pad_token_id, -100
        )
        prompt_len = len(self.blip.tokenizer(prompt_prefix).input_ids) - 1
        decoder_targets[:, :prompt_len] = -100

        image_atts = torch.ones(
            image_embeds_enhanced.size()[:-1], dtype=torch.long, device=device
        )

        decoder_out = self.blip.text_decoder(
            input_ids,
            attention_mask=text_inputs.attention_mask,
            encoder_hidden_states=image_embeds_enhanced,
            encoder_attention_mask=image_atts,
            labels=decoder_targets,
            return_dict=True,
        )
        ce_loss = decoder_out.loss

        # Safely retrieve prediction logits across different BLIP med.py implementations
        if hasattr(decoder_out, "logits") and decoder_out.logits is not None:
            logits = decoder_out.logits[:, :-1, :]
        else:
            logits = self.blip.text_decoder(
                input_ids,
                attention_mask=text_inputs.attention_mask,
                encoder_hidden_states=image_embeds_enhanced,
                encoder_attention_mask=image_atts,
                return_logits=True,
            )

        target_ids = input_ids[:, 1:]
        valid_mask = (decoder_targets[:, 1:] != -100).float()

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        seq_log_probs = (token_log_probs * valid_mask).sum(dim=-1) / (
            valid_mask.sum(dim=-1) + 1e-6
        )

        return ce_loss, seq_log_probs

    def compute_visual_critic_similarity(
        self, image_embeds_calibrated: torch.Tensor, captions: List[str]
    ) -> torch.Tensor:
        """
        Computes the differentiable cross-modal cosine alignment score S_vis(c_i) (Eq. 12)
        between generated/ground-truth captions and ViPER-calibrated visual features.
        """
        device = image_embeds_calibrated.device
        text_inputs = self.blip.tokenizer(
            captions, padding=True, truncation=True, max_length=35, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            text_out = self.blip.text_decoder.bert(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                return_dict=True,
                encoder_hidden_states=None,
            )
            text_embeds = text_out.last_hidden_state[:, 0, :].unsqueeze(1)

        reconstructed_vis = self.adapter(image_embeds_calibrated, text_embeds)
        cos_sim = F.cosine_similarity(
            reconstructed_vis.squeeze(1), text_embeds.squeeze(1), dim=-1
        )
        return cos_sim

    @torch.no_grad()
    def sample_and_baseline_decode(
        self,
        image_embeds_calibrated: torch.Tensor,
        image_embeds_enhanced: torch.Tensor,
    ) -> Tuple[List[str], List[str]]:
        """
        Generates:
          1. Self-critical baseline captions via deterministic greedy decoding.
          2. Exploratory sampled captions via multinomial nucleus sampling for SCST.
        """
        batch_size = image_embeds_enhanced.size(0)
        device = image_embeds_enhanced.device
        prompts = [self.blip.prompt] * batch_size
        prompt_inputs = self.blip.tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        input_ids = prompt_inputs.input_ids.clone()
        input_ids[:, 0] = self.blip.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]
        image_atts = torch.ones(
            image_embeds_enhanced.size()[:-1], dtype=torch.long, device=device
        )

        # A. Baseline trajectory via deterministic greedy decoding
        greedy_out = self.blip.text_decoder.generate(
            input_ids=input_ids,
            encoder_hidden_states=image_embeds_enhanced,
            encoder_attention_mask=image_atts,
            max_length=22,
            min_length=8,
            num_beams=1,
            do_sample=False,
            eos_token_id=self.blip.tokenizer.sep_token_id,
            pad_token_id=self.blip.tokenizer.pad_token_id,
        )
        baseline_decoded = self.blip.tokenizer.batch_decode(greedy_out, skip_special_tokens=True)
        baseline_captions = [
            strip_blip_prompt(c, self.blip.prompt) for c in baseline_decoded
        ]

        # B. Exploratory trajectory via multinomial nucleus sampling
        sample_out = self.blip.text_decoder.generate(
            input_ids=input_ids,
            encoder_hidden_states=image_embeds_enhanced,
            encoder_attention_mask=image_atts,
            max_length=22,
            min_length=8,
            num_beams=1,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
            eos_token_id=self.blip.tokenizer.sep_token_id,
            pad_token_id=self.blip.tokenizer.pad_token_id,
        )
        sampled_decoded = self.blip.tokenizer.batch_decode(sample_out, skip_special_tokens=True)
        sampled_captions = [
            strip_blip_prompt(c, self.blip.prompt) for c in sampled_decoded
        ]

        return baseline_captions, sampled_captions


# ==============================================================================
# 3. CIDEr + Visual Entailment Reward Computation (SCST)
# ==============================================================================
def compute_scst_cider_rewards(
    cider_scorer: Cider,
    image_ids: List[str],
    baseline_captions: List[str],
    sampled_captions: List[str],
    references_dict: Dict[str, List[str]],
    device: torch.device,
) -> torch.Tensor:
    """Computes the self-critical CIDEr advantage: A = CIDEr(sampled) - CIDEr(baseline)."""
    gts: Dict[str, List[str]] = {}
    res_sample: Dict[str, List[str]] = {}
    res_base: Dict[str, List[str]] = {}

    for uid, base_cap, samp_cap in zip(image_ids, baseline_captions, sampled_captions):
        gts[uid] = references_dict[uid]
        res_sample[uid] = [samp_cap if samp_cap else "a photo"]
        res_base[uid] = [base_cap if base_cap else "a photo"]

    _, scores_sample = cider_scorer.compute_score(gts, res_sample)
    _, scores_base = cider_scorer.compute_score(gts, res_base)

    advantages = [float(s - b) for s, b in zip(scores_sample, scores_base)]
    return torch.tensor(advantages, dtype=torch.float32, device=device)


# ==============================================================================
# 4. Main Training Loop (Differential LR, AMP, 4-Step Grad Accumulation)
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-3 Joint ViPER (\alpha) & CIDEr SCST Optimization for VeriVLM-Agent"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="./pretrained/model_base_caption_capfilt_large.pth",
        help="Path to the pre-trained BLIP base checkpoint.",
    )
    parser.add_argument(
        "--stage1_adapter_ckpt",
        type=str,
        default="./checkpoints/checkpoints_beta/adapter_epoch_10.pth",
        help="Path to the converged Stage-1 Beta-SEA adapter checkpoint.",
    )
    parser.add_argument(
        "--img_root",
        type=str,
        default="./data/coco/train2014",
        help="Directory containing MS COCO training images.",
    )
    parser.add_argument(
        "--ann_json_path",
        type=str,
        default="./data/coco/annotations/captions_train2014.json",
        help="Path to MS COCO training caption annotations JSON.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/checkpoints_stage3_viper",
        help="Directory to save jointly optimized ViPER + Beta-SEA checkpoints.",
    )

    # Exact hyperparameters from Section 4.2 & Section 4.7 of the paper
    parser.add_argument(
        "--mode",
        type=str,
        default="scst",
        choices=["scst", "ce"],
        help="Optimization mode: 'scst' (CIDEr RL optimization, Table 2) or 'ce' (Cross-Entropy, Table 3).",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Physical batch size (Section 4.2: 16).")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Grad accumulation (Section 4.2: 4 -> eff 64).")
    parser.add_argument("--epochs", type=int, default=10, help="Total training epochs (Section 4.2: 10).")
    parser.add_argument("--lr_viper", type=float, default=1e-4, help="Differential LR for ViPER scalar (1e-4).")
    parser.add_argument("--lr_adapter", type=float, default=2e-4, help="Differential LR for Beta-SEA adapter (2e-4).")
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    # ViPER & Gamma-DHR fixed boundary hyperparameters
    parser.add_argument("--warm_start_scale", type=float, default=0.02, help="Initial effective scale \sigma(\alpha)=0.02.")
    parser.add_argument("--lambda_max", type=float, default=0.5, help="Upper bound \lambda=0.5 in Eq. (3).")
    parser.add_argument("--gamma", type=float, default=0.35, help="Optimal Gamma-DHR visual weight from Stage 2.")
    parser.add_argument("--reg_weight", type=float, default=0.1, help="Controlled residual manifold regularization weight.")

    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--max_samples", type=int, default=0, help="Limit dataset size for fast debugging (0=all).")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    logger.info(
        "Starting Stage-3 Joint Optimization (Mode=%s | Physical BS=%d | Grad Accum=%d | Effective BS=%d)",
        args.mode.upper(),
        args.batch_size,
        args.grad_accum_steps,
        args.batch_size * args.grad_accum_steps,
    )

    dataset = Stage3COCODataset(
        img_root=args.img_root,
        ann_json_path=args.ann_json_path,
        max_samples=args.max_samples,
    )
    collator = Stage3Collator(transform=build_transform(args.image_size))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )

    model = BetaBLIPStage3Model(
        base_model_path=args.base_model_path,
        stage1_adapter_ckpt=args.stage1_adapter_ckpt,
        image_size=args.image_size,
        warm_start_scale=args.warm_start_scale,
        lambda_max=args.lambda_max,
    ).to(device)

    # Differential learning rate setup (Section 4.2: 1e-4 for ViPER scalar, 2e-4 for Beta-SEA adapter)
    optimizer = optim.AdamW(
        [
            {
                "params": list(model.viper.parameters()),
                "lr": args.lr_viper,
                "weight_decay": 0.0,
                "name": "viper_scalar",
            },
            {
                "params": list(model.adapter.parameters()),
                "lr": args.lr_adapter,
                "weight_decay": args.weight_decay,
                "name": "beta_sea_adapter",
            },
        ]
    )

    total_update_steps = max(1, (len(dataloader) * args.epochs) // args.grad_accum_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_update_steps, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    cider_scorer = Cider() if args.mode == "scst" else None

    trajectory_log: List[Dict[str, float]] = [
        {"epoch": 0, "effective_alpha": round(float(model.viper.get_effective_scale().item()), 4)}
    ]
    logger.info(
        "Epoch 0 Initial Warm-Start Effective Scale \\sigma(\\alpha) = %.4f",
        trajectory_log[0]["effective_alpha"],
    )

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(
            dataloader, desc=f"Stage-3 [{args.mode.upper()}] Epoch {epoch + 1}/{args.epochs}"
        )
        for step, batch in enumerate(pbar):
            if batch is None:
                continue

            images = batch["images"].to(device, non_blocking=True)
            image_ids = batch["image_ids"]
            primary_captions = batch["primary_captions"]
            references_dict = batch["references_dict"]

            with torch.amp.autocast("cuda", enabled=use_amp):
                img_calibrated, img_enhanced = model.encode_and_inject_prior(images)

                if args.mode == "ce":
                    ce_loss, _ = model.compute_sequence_log_probs(img_enhanced, primary_captions)
                    vis_sim = model.compute_visual_critic_similarity(
                        img_calibrated, primary_captions
                    )
                    align_reg = (1.0 - vis_sim).mean()
                    task_loss = ce_loss + 0.5 * align_reg
                else:
                    base_caps, samp_caps = model.sample_and_baseline_decode(
                        img_calibrated,
                        img_enhanced,
                    )
                    cider_advantages = compute_scst_cider_rewards(
                        cider_scorer=cider_scorer,
                        image_ids=image_ids,
                        baseline_captions=base_caps,
                        sampled_captions=samp_caps,
                        references_dict=references_dict,
                        device=device,
                    )

                    _, seq_log_probs = model.compute_sequence_log_probs(img_enhanced, samp_caps)
                    vis_sim = model.compute_visual_critic_similarity(img_calibrated, samp_caps)

                    scst_loss = -(cider_advantages.detach() * seq_log_probs).mean()
                    vis_critic_loss = (1.0 - vis_sim).mean()
                    task_loss = scst_loss + args.gamma * vis_critic_loss

                # Controlled residual regularization preventing over-sharpening (Section 4.7, Fig. 10b)
                eff_alpha = model.viper.get_effective_scale()
                residual_reg = args.reg_weight * F.smooth_l1_loss(
                    eff_alpha,
                    torch.tensor(model.equilibrium_target, device=device, dtype=eff_alpha.dtype),
                )

                loss = (task_loss + residual_reg) / args.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            current_loss = loss.item() * args.grad_accum_steps
            epoch_loss += current_loss
            curr_alpha_val = float(model.viper.get_effective_scale().item())
            pbar.set_postfix(
                {
                    "Loss": f"{current_loss:.4f}",
                    "Scale_a": f"{curr_alpha_val:.4f}",
                    "LR_v": f"{optimizer.param_groups[0]['lr']:.1e}",
                }
            )

        final_epoch_alpha = round(float(model.viper.get_effective_scale().item()), 4)
        avg_loss = epoch_loss / max(1, len(dataloader))
        trajectory_log.append(
            {
                "epoch": epoch + 1,
                "avg_loss": round(avg_loss, 4),
                "effective_alpha": final_epoch_alpha,
            }
        )

        # Save unified checkpoint compatible with FullStackBLIPEvaluator
        ckpt_payload = {
            "epoch": epoch + 1,
            "model_state_dict": model.adapter.state_dict(),
            "viper_state_dict": model.viper.state_dict(),
            "effective_viper_scale": final_epoch_alpha,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "trajectory_log": trajectory_log,
        }
        save_path = os.path.join(args.output_dir, f"adapter_epoch_{epoch + 1}.pth")
        torch.save(ckpt_payload, save_path)
        logger.info(
            "Epoch %d saved -> %s | Avg Loss: %.4f | Effective ViPER Scale \\sigma(\\alpha): %.4f",
            epoch + 1,
            save_path,
            avg_loss,
            final_epoch_alpha,
        )

    dynamics_json = os.path.join(args.output_dir, "viper_alpha_dynamics.json")
    with open(dynamics_json, "w", encoding="utf-8") as f:
        json.dump(trajectory_log, f, indent=2)
    logger.info("Stage-3 optimization finished. Alpha trajectory saved to: %s", dynamics_json)


if __name__ == "__main__":
    main()