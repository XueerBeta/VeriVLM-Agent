# -*- coding: utf-8 -*-
"""
Stage-1 Training Script: Beta-Semantic Entailment Adapter (Beta-SEA).
Freezes the pre-trained BLIP backbone and optimizes the 5M cross-attention adapter
under the asymmetric Beta-Contextual Alignment Loss (Beta-CAL).
"""

import argparse
import logging
import os
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from models.blip import blip_decoder
except ImportError as exc:
    raise ImportError(
        "Cannot import 'models.blip'. Please ensure this script is executed from the project root."
    ) from exc

from core.modules import BetaSemanticAdapter
from data_pipeline.syntactic_dataset import (
    ensure_nltk_resources,
    init_blip_tokenizer,
    build_transform,
    BetaCOCODataset,
    BlipCollate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BetaSEA-Train")


class BetaCALLoss(nn.Module):
    """
    Beta-Contextual Alignment Loss (Beta-CAL).
    Assigns a soft target of beta (0.5) to intra-image local phrase queries and 1.0 to
    exact diagonal correspondences, preventing overconfident local alignment and context bleeding.
    """

    def __init__(self, beta: float = 0.5, temperature: float = 0.07) -> None:
        super().__init__()
        self.beta = beta
        self.temperature = temperature

    def forward(
        self,
        visual_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        text_mapping: List[int],
    ) -> torch.Tensor:
        """
        Args:
            visual_embeds: Adapter-refined visual features [Total_Queries, Embed_Dim].
            text_embeds: Textual query embeddings [Total_Queries, Embed_Dim].
            text_mapping: Number of hierarchical queries per image in the batch.

        Returns:
            Symmetric soft-target cross-modal alignment loss scalar.
        """
        v_flat = F.normalize(visual_embeds, p=2, dim=-1)
        t_flat = F.normalize(text_embeds, p=2, dim=-1)

        logits = torch.matmul(v_flat, t_flat.t()) / self.temperature
        num_queries = logits.shape[0]
        labels = torch.zeros((num_queries, num_queries), device=logits.device)

        start_idx = 0
        for count in text_mapping:
            end_idx = start_idx + count
            labels[start_idx:end_idx, start_idx:end_idx] = self.beta
            start_idx = end_idx

        labels.fill_diagonal_(1.0)
        target_probs = labels / (labels.sum(dim=1, keepdim=True) + 1e-8)

        loss_v2t = -torch.sum(target_probs * F.log_softmax(logits, dim=1), dim=1).mean()
        loss_t2v = -torch.sum(target_probs * F.log_softmax(logits.t(), dim=1), dim=1).mean()

        return 0.5 * (loss_v2t + loss_t2v)


class BetaBLIPAligner(nn.Module):
    """
    Alignment wrapper that freezes the 224M BLIP backbone (ViT + BERT) and trains
    only the lightweight BetaSemanticAdapter (~5M parameters).
    """

    def __init__(self, base_model_path: str, image_size: int = 384, vit_type: str = "base") -> None:
        super().__init__()
        logger.info("Loading custom BLIP baseline from: %s", base_model_path)
        self.blip = blip_decoder(pretrained=base_model_path, image_size=image_size, vit=vit_type)

        logger.info("Initializing Beta-Semantic Entailment Adapter (Beta-SEA)...")
        self.adapter = BetaSemanticAdapter(embed_dim=768)

        for param in self.blip.parameters():
            param.requires_grad = False

        for param in self.adapter.parameters():
            param.requires_grad = True

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_mapping: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts frozen visual/textual representations and projects visual features
        into the query-conditioned alignment manifold via Beta-SEA.
        """
        with torch.no_grad():
            image_embeds = self.blip.visual_encoder(images)
            text_outputs = self.blip.text_decoder.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            text_embeds = text_outputs.last_hidden_state[:, 0, :]

        expanded_img_embeds = [
            image_embeds[i].unsqueeze(0).expand(count, -1, -1)
            for i, count in enumerate(text_mapping)
        ]
        aligned_img_inputs = torch.cat(expanded_img_embeds, dim=0)
        text_queries = text_embeds.unsqueeze(1)

        visual_focused = self.adapter(aligned_img_inputs, text_queries).squeeze(1)
        return visual_focused, text_embeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-1 Training: Beta-SEA Multi-Granular Adapter Alignment"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="./pretrained/model_base_caption_capfilt_large.pth",
        help="Path to the pre-trained BLIP .pth checkpoint.",
    )
    parser.add_argument(
        "--img_root",
        type=str,
        default="./data/coco/train2014",
        help="Root directory of MS COCO training images.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="./data/coco/annotations/coco_karpathy_train_phrases.json",
        help="Path to pre-processed Karpathy phrase annotations JSON.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints/checkpoints_beta",
        help="Directory to save trained adapter checkpoints.",
    )

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.5, help="Asymmetric soft-label target.")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max_phrases", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Pause for user confirmation between epochs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    ensure_nltk_resources()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    latest_ckpt_path = os.path.join(args.save_dir, "checkpoint_latest.pth")

    logger.info("Executing Stage-1 Beta-SEA training on device: %s", device)

    tokenizer = init_blip_tokenizer()
    dataset = BetaCOCODataset(
        img_folder=args.img_root,
        ann_file=args.json_path,
        max_phrases=args.max_phrases,
        max_samples=args.max_samples,
    )
    collate_fn = BlipCollate(tokenizer=tokenizer, transform=build_transform(args.image_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )

    model = BetaBLIPAligner(base_model_path=args.base_model_path, image_size=args.image_size).to(
        device
    )
    loss_fn = BetaCALLoss(beta=args.beta, temperature=args.temperature).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(
        "Trainable Adapter Parameters: %.2fM",
        sum(p.numel() for p in trainable_params) / 1e6,
    )

    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_steps = max(1, (len(loader) * args.epochs) // args.accum_steps)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.min_lr
    )

    start_epoch = 0
    if os.path.exists(latest_ckpt_path):
        logger.info("Resuming training state from checkpoint: %s", latest_ckpt_path)
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.adapter.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info("Resumed successfully. Continuing from Epoch %d.", start_epoch + 1)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Stage-1 Epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            if batch is None:
                continue

            images = batch["images"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            mapping = batch["text_mapping"]

            with torch.amp.autocast("cuda", enabled=use_amp):
                vis_embeds, txt_embeds = model(images, input_ids, mask, mapping)
                loss = loss_fn(vis_embeds, txt_embeds, mapping)
                scaled_loss = loss / args.accum_steps

            scaler.scale(scaled_loss).backward()

            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            current_loss = loss.item()
            total_loss += current_loss
            curr_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"Loss": f"{current_loss:.4f}", "LR": f"{curr_lr:.2e}"})

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.adapter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }

        epoch_save_path = os.path.join(args.save_dir, f"adapter_epoch_{epoch + 1}.pth")
        torch.save(checkpoint_data, epoch_save_path)
        torch.save(checkpoint_data, latest_ckpt_path)

        avg_loss = total_loss / max(1, len(loader))
        logger.info(
            "Epoch %d completed | Avg Loss: %.4f | Saved: %s",
            epoch + 1,
            avg_loss,
            epoch_save_path,
        )

        if args.interactive and epoch < args.epochs - 1:
            input(f"Press [Enter] to continue to Epoch {epoch + 2}...")

    logger.info("All Stage-1 training epochs completed successfully.")


if __name__ == "__main__":
    main()