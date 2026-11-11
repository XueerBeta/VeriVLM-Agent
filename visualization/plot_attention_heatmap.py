# -*- coding: utf-8 -*-
"""
Visualization script for VeriVLM-Agent (Beta-BLIP).
Generates fine-grained cross-modal grounding attention heatmaps (reproducing Fig. 4 in the paper)
by computing strict contrastive cross-attention between ViPER-refined visual tokens
and Beta-SEA text queries.
"""

import argparse
import logging
import os
from typing import Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

try:
    from models.blip import blip_decoder
except ImportError as exc:
    raise ImportError(
        "Cannot import 'models.blip'. Please run this script from the project root directory."
    ) from exc

from core.modules import ViPER, BetaSemanticAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("VeriVLM-Visualizer")


class FullStackAttentionVisualizer(nn.Module):
    """
    Evaluator wrapper designed to capture raw cross-attention weights from Beta-SEA
    conditioned on ViPER-calibrated visual features.
    """

    def __init__(
            self,
            base_model_path: str,
            image_size: int = 384,
            vit_type: str = "base",
            viper_scale: float = 0.083,
    ) -> None:
        super().__init__()
        logger.info("Initializing Visualizer | ViT: %s | ViPER Scale: %.3f", vit_type, viper_scale)
        self.blip = blip_decoder(pretrained=base_model_path, image_size=image_size, vit=vit_type)
        self.viper = ViPER(scale=viper_scale, trainable=False, use_variance_routing=True)
        self.adapter = BetaSemanticAdapter(embed_dim=768)

        # Hook into MultiheadAttention to intercept raw attention weights
        self._register_attention_hook()

        for param in self.parameters():
            param.requires_grad = False

    def _register_attention_hook(self) -> None:
        """Hooks into Beta-SEA cross-attention layer to extract attention tensors."""
        original_forward = self.adapter.cross_attn.forward

        def hooked_forward(*args, **kwargs):
            # Force PyTorch MultiheadAttention to return average attention weights
            kwargs["average_attn_weights"] = False
            out = original_forward(*args, **kwargs)
            # out[1] represents attention weights of shape [Batch, Num_Heads, Target_Len, Source_Len]
            self.raw_attn_weights = out[1].detach().cpu()
            return out

        self.adapter.cross_attn.forward = hooked_forward
        self.raw_attn_weights: Optional[torch.Tensor] = None

    @torch.no_grad()
    def extract_contrastive_heatmap(
            self,
            image_tensor: torch.Tensor,
            prompt_query: str,
            target_word: str,
    ) -> np.ndarray:
        """
        Computes strict contrastive attention (target word attention minus global sentence average)
        to isolate fine-grained visual grounding.
        """
        tokenizer = self.blip.tokenizer
        prompt_inputs = tokenizer([prompt_query], return_tensors="pt", padding=True).to(image_tensor.device)

        # 1. ViT Visual Encoding
        image_embeds_raw = self.blip.visual_encoder(image_tensor)

        # 2. Module A - ViPER Perceptual Calibration
        image_embeds_viper = self.viper(image_embeds_raw)

        # 3. Text Encoding
        text_outputs = self.blip.text_decoder.bert(
            input_ids=prompt_inputs.input_ids,
            attention_mask=prompt_inputs.attention_mask,
            return_dict=True,
        )
        text_queries = text_outputs.last_hidden_state

        # 4. Module B - Beta-SEA Cross-Modal Alignment (triggers attention hook)
        _ = self.adapter(image_embeds_viper, text_queries)

        if self.raw_attn_weights is None:
            raise RuntimeError("Failed to capture cross-attention weights from Beta-SEA.")

        # Parse token positions
        input_ids = prompt_inputs.input_ids[0].cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        if target_word not in tokens:
            raise ValueError(
                f"Target word '{target_word}' not found in tokenized prompt tokens: {tokens}. "
                "Please ensure the target word matches the tokenizer vocabulary."
            )
        target_idx = tokens.index(target_word)

        # Extract attention weights: shape [Num_Heads, Target_Len, Source_Len]
        # Exclude the first spatial patch if it corresponds to the [CLS] token (slice 1:)
        all_weights = self.raw_attn_weights[0, :, :, 1:].numpy()

        target_attn = all_weights[:, target_idx, :]  # [Num_Heads, Num_Patches]
        sentence_avg_attn = np.mean(all_weights, axis=1)  # [Num_Heads, Num_Patches]

        # Strict Contrastive Attention (Eq. in visual grounding analysis)
        contrastive_attn = target_attn - sentence_avg_attn
        contrastive_attn = np.maximum(contrastive_attn, 0.0)

        # Aggregate across attention heads and square for sharper spatial localization
        word_attention = np.max(contrastive_attn, axis=0)
        word_attention = word_attention ** 2.0
        return word_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Cross-Modal Grounding Attention Heatmaps for VeriVLM-Agent"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./assets/demo_example.jpg",
        help="Path to input test image.",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="./pretrained/model_base_caption_capfilt_large.pth",
        help="Path to pre-trained BLIP base checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints/checkpoints_beta/adapter_epoch_10.pth",
        help="Path to trained Beta-SEA adapter checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/heatmaps",
        help="Directory to save generated heatmap overlays.",
    )

    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--viper_scale", type=float, default=0.083, help="Optimal ViPER scale alpha.")
    parser.add_argument("--phrase_query", type=str, default="a baby", help="Full text prompt query.")
    parser.add_argument("--target_word", type=str, default="baby", help="Specific target word to isolate.")

    parser.add_argument("--shared_alpha", type=float, default=0.45, help="Heatmap overlay blending alpha.")
    parser.add_argument("--blur_kernel", type=int, default=101, help="Gaussian blur kernel size (odd integer).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.image_path):
        logger.error("Input image not found: %s", args.image_path)
        return

    visualizer = FullStackAttentionVisualizer(
        base_model_path=args.base_model_path,
        image_size=args.image_size,
        vit_type="base",
        viper_scale=args.viper_scale,
    ).to(device)

    if os.path.exists(args.checkpoint_path):
        logger.info("Loading adapter weights from: %s", args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        visualizer.adapter.load_state_dict(state_dict, strict=False)
    else:
        logger.warning("Adapter checkpoint not found at %s; using uninitialized adapter weights.", args.checkpoint_path)

    visualizer.eval()

    raw_image = Image.open(args.image_path).convert("RGB")
    raw_image_resized = np.array(raw_image.resize((args.image_size, args.image_size)))

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    image_tensor = transform(raw_image).unsqueeze(0).to(device)

    logger.info("Extracting strict contrastive attention for query: '%s' (Target: '%s')", args.phrase_query,
                args.target_word)
    word_attention = visualizer.extract_contrastive_heatmap(
        image_tensor=image_tensor,
        prompt_query=args.phrase_query,
        target_word=args.target_word,
    )

    # Reshape patch scores into 2D spatial grid
    num_patches = word_attention.shape[0]
    grid_size = int(np.sqrt(num_patches))
    w_map = word_attention.reshape(grid_size, grid_size).astype(np.float32)

    # Smoothing and colormap rendering
    w_map = cv2.GaussianBlur(w_map, (3, 3), 0)
    w_map_resized = cv2.resize(w_map, (args.image_size, args.image_size), interpolation=cv2.INTER_CUBIC)

    kernel_size = args.blur_kernel if args.blur_kernel % 2 == 1 else args.blur_kernel + 1
    w_map_blurred = cv2.GaussianBlur(w_map_resized, (kernel_size, kernel_size), 0)

    w_map_norm = (w_map_blurred - w_map_blurred.min()) / (w_map_blurred.max() - w_map_blurred.min() + 1e-8)

    heatmap_colored = np.uint8(255 * w_map_norm)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Fixed syntax: alpha blending with 1.0 - shared_alpha
    overlay = cv2.addWeighted(
        raw_image_resized, 1.0 - args.shared_alpha, heatmap_colored, args.shared_alpha, 0.0
    )

    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    output_filename = os.path.join(args.output_dir, f"{base_name}_heatmap_{args.target_word}.png")
    plt.imsave(output_filename, overlay)
    logger.info("Heatmap successfully saved to: %s", output_filename)


if __name__ == "__main__":
    main()