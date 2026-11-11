# -*- coding: utf-8 -*-
"""
Command-line entry point for VeriVLM-Agent (Beta-BLIP) single-image inference.
Supports both standard single-pass full-stack inference and LangGraph self-reflective Agent mode.
"""

import argparse
import logging
import os
import sys
import torch
from PIL import Image

from core.evaluator import FullStackBLIPEvaluator
from agent.verivlm_graph import VeriVLMAgentWorkflow
from data_pipeline.syntactic_dataset import build_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("VeriVLM-CLI")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for inference configuration."""
    parser = argparse.ArgumentParser(
        description="VeriVLM-Agent: Closed-Loop Semantic Calibration & Self-Reflective VLM Inference"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./assets/demo_example.jpg",
        help="Path to the input test image.",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="./pretrained/model_base_caption_capfilt_large.pth",
        help="Path to the pre-trained BLIP base checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/checkpoints_beta",
        help="Directory containing trained Beta-SEA adapter checkpoints.",
    )
    parser.add_argument("--epoch", type=int, default=10, help="Adapter checkpoint epoch index to load.")
    parser.add_argument("--vit_type", type=str, default="base", choices=["base", "large"])
    parser.add_argument("--image_size", type=int, default=384)

    # Optimal hyperparameters from Section 4.7 of the paper
    parser.add_argument("--viper_scale", type=float, default=0.083, help="ViPER modulation scale alpha.")
    parser.add_argument("--gamma", type=float, default=0.35, help="Gamma-DHR visual reranking weight.")
    parser.add_argument("--diversity_penalty", type=float, default=1.45, help="Grouped beam penalty lambda.")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Sequence length penalty factor.")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of decoding beams.")

    # Execution mode & Agent parameters
    parser.add_argument(
        "--mode",
        type=str,
        default="agent",
        choices=["standard", "agent"],
        help="Execution mode: 'standard' (single pass) or 'agent' (LangGraph ReAct loop).",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=0.25,
        help="Minimum raw cosine alignment score required before triggering localized crop tool.",
    )
    parser.add_argument(
        "--dpo_export_path",
        type=str,
        default="./outputs/dpo_harvested_pairs.jsonl",
        help="Path to save automatically mined DPO preference pairs.",
    )
    return parser.parse_args()


def print_evaluation_report(result: dict, mode: str) -> None:
    """Formats and prints a clean, academic-grade diagnostic report."""
    separator = "=" * 76
    sub_sep = "-" * 76
    print(f"\n{separator}")
    print(f"VERIVLM-AGENT INFERENCE REPORT (Mode: {mode.upper()})")
    print(sub_sep)
    print(f"Optimal Caption     : {result['best_caption']}")
    print(f"Final Combined Score: {result['best_s_final']:.4f}")
    print(f"Visual Cosine Score : {result['best_s_vis_raw']:.4f}")
    print(f"Salient Region BBox : {result['salient_bbox']}")
    print(sub_sep)
    print("Candidate Hypotheses Breakdown (Gamma-DHR Dual-Stream Verification):")
    for cand in result["candidates"]:
        print(
            f"  [Beam {cand['rank_index']}] S_final={cand['s_final']:.4f} "
            f"(S_lang={cand['s_lang_norm']:.4f}, S_vis={cand['s_vis_norm']:.4f}) -> {cand['caption']}"
        )
    if result.get("dpo_pair"):
        print(sub_sep)
        print(f"DPO Hard Negative Mined -> Rejected: '{result['dpo_pair']['rejected']}'")
    print(f"{separator}\n")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.image_path):
        logger.error("Input image file not found: %s", args.image_path)
        sys.exit(1)

    ckpt_path = os.path.join(args.checkpoint_dir, f"adapter_epoch_{args.epoch}.pth")
    if not os.path.exists(ckpt_path):
        logger.error("Adapter checkpoint file not found: %s", ckpt_path)
        sys.exit(1)

    if args.dpo_export_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.dpo_export_path)), exist_ok=True)

    evaluator = FullStackBLIPEvaluator(
        base_model_path=args.base_model_path,
        image_size=args.image_size,
        vit_type=args.vit_type,
        viper_scale=args.viper_scale,
    ).to(device)
    evaluator.load_adapter_checkpoint(ckpt_path, device=device)
    evaluator.eval()

    transform = build_transform(args.image_size)

    if args.mode == "standard":
        raw_image = Image.open(args.image_path).convert("RGB")
        image_tensor = transform(raw_image).unsqueeze(0).to(device)
        result = evaluator.generate_full_stack(
            image=image_tensor,
            gamma=args.gamma,
            num_beams=args.num_beams,
            diversity_penalty=args.diversity_penalty,
            length_penalty=args.length_penalty,
            dpo_export_path=args.dpo_export_path,
        )
        print_evaluation_report(result, mode="standard")
    else:
        agent = VeriVLMAgentWorkflow(
            evaluator=evaluator,
            transform=transform,
            device=device,
            gamma=args.gamma,
            num_beams=args.num_beams,
            diversity_penalty=args.diversity_penalty,
            length_penalty=args.length_penalty,
            dpo_export_path=args.dpo_export_path,
        )
        final_state = agent.invoke(image_path=args.image_path, vis_threshold=args.vis_threshold)
        print_evaluation_report(final_state, mode="agent")


if __name__ == "__main__":
    main()