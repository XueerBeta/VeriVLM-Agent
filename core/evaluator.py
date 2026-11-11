# -*- coding: utf-8 -*-
"""
Full-stack closed-loop inference evaluator integrating ViPER, Beta-SEA,
and Gamma-Diversity Hybrid Reranking (Gamma-DHR).
"""

import json
import logging
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn

try:
    from models.blip import blip_decoder
except ImportError as exc:
    raise ImportError(
        "Cannot import 'models.blip'. Please execute the script from the project root directory."
    ) from exc

from core.modules import ViPER, BetaSemanticAdapter

logger = logging.getLogger(__name__)


def strip_blip_prompt(text: str, prompt: str) -> str:
    """Safely removes the BLIP prefix prompt from a decoded caption string."""
    clean_text = text.strip()
    clean_prompt = prompt.strip()
    if clean_prompt and clean_text.lower().startswith(clean_prompt.lower()):
        return clean_text[len(clean_prompt) :].strip()
    return clean_text


class FullStackBLIPEvaluator(nn.Module):
    """
    Closed-loop vision-language evaluator implementing:
    1. ViPER perceptual feature calibration.
    2. Beta-SEA feed-forward task prior injection.
    3. Grouped Beam Search with inter-group diversity penalties.
    4. Gamma-DHR dual-stream posterior verification and DPO pair harvesting.
    """

    def __init__(
        self,
        base_model_path: str,
        image_size: int = 384,
        vit_type: str = "base",
        viper_scale: float = 0.083,
        use_variance_routing: bool = True,
    ) -> None:
        super().__init__()
        logger.info(
            "Initializing BLIP backbone (ViT: %s, ViPER scale: %.4f)", vit_type, viper_scale
        )
        self.blip = blip_decoder(pretrained=base_model_path, image_size=image_size, vit=vit_type)
        self.viper = ViPER(
            scale=viper_scale, trainable=False, use_variance_routing=use_variance_routing
        )
        self.adapter = BetaSemanticAdapter(embed_dim=768)

        for param in self.parameters():
            param.requires_grad = False

    def load_adapter_checkpoint(
        self, checkpoint_path: str, device: torch.device, override_viper_from_ckpt: bool = False
    ) -> None:
        """
        Loads trained Beta-SEA adapter weights from disk.
        Compatible with both Stage-1 and Stage-3 checkpoints.
        """
        logger.info("Loading Beta-SEA adapter checkpoint from: %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected = self.adapter.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys when loading adapter state_dict: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys when loading adapter state_dict: %s", unexpected)

        if override_viper_from_ckpt and isinstance(checkpoint, dict):
            if "effective_viper_scale" in checkpoint:
                ckpt_scale = float(checkpoint["effective_viper_scale"])
                self.viper.set_fixed_scale(ckpt_scale)
                logger.info("Updated ViPER scale from Stage-3 checkpoint: %.4f", ckpt_scale)

    @torch.no_grad()
    def generate_full_stack(
        self,
        image: torch.Tensor,
        gamma: float = 0.35,
        num_beams: int = 5,
        diversity_penalty: float = 1.45,
        length_penalty: float = 1.0,
        max_length: int = 25,
        min_length: int = 12,
        dpo_export_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes closed-loop inference and returns both the optimal caption and
        diagnostic metrics for downstream Agent reflection and DPO data synthesis.
        """
        # Step 1: Visual encoding via frozen ViT backbone
        image_embeds_raw = self.blip.visual_encoder(image)

        # Step 2: Module A - ViPER perceptual feature calibration & bbox extraction
        image_embeds = self.viper(image_embeds_raw)
        salient_bbox = self.viper.extract_high_density_bbox(image_embeds_raw)[0]

        # Step 3: Module B - Beta-SEA feed-forward task prior injection (Eq. 9)
        prompt = [self.blip.prompt] * image.size(0)
        prompt_inputs = self.blip.tokenizer(prompt, return_tensors="pt", padding=True).to(
            image.device
        )

        text_outputs = self.blip.text_decoder.bert(
            input_ids=prompt_inputs.input_ids,
            attention_mask=prompt_inputs.attention_mask,
            return_dict=True,
            encoder_hidden_states=None,
        )
        text_queries = text_outputs.last_hidden_state[:, 0, :].unsqueeze(1)
        enhanced_feat = self.adapter(image_embeds, text_queries)

        image_embeds_enhanced = image_embeds.clone()
        image_embeds_enhanced[:, 0, :] = image_embeds_enhanced[:, 0, :] + enhanced_feat.squeeze(1)
        image_atts = torch.ones(
            image_embeds_enhanced.size()[:-1], dtype=torch.long, device=image.device
        )

        # Step 4: Module C - Gamma-DHR diversity-guided grouped beam search (Eq. 10)
        input_ids = prompt_inputs.input_ids.clone()
        input_ids[:, 0] = self.blip.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        use_diversity = diversity_penalty > 0.0
        outputs = self.blip.text_decoder.generate(
            input_ids=input_ids,
            encoder_hidden_states=image_embeds_enhanced,
            encoder_attention_mask=image_atts,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            num_beam_groups=num_beams if use_diversity else 1,
            diversity_penalty=diversity_penalty if use_diversity else 0.0,
            repetition_penalty=1.0,
            eos_token_id=self.blip.tokenizer.sep_token_id,
            pad_token_id=self.blip.tokenizer.pad_token_id,
            num_return_sequences=num_beams,
            return_dict_in_generate=True,
            output_scores=True,
            length_penalty=length_penalty,
        )

        # Step 5: Module C - Gamma-DHR dual-stream posterior verification (Eq. 11-13)
        lang_scores = outputs.sequences_scores
        decoded_seqs = self.blip.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        candidates: List[str] = [
            strip_blip_prompt(seq, self.blip.prompt) for seq in decoded_seqs
        ]

        text_inputs = self.blip.tokenizer(
            candidates, return_tensors="pt", padding=True, truncation=True
        ).to(image.device)

        cand_text_outputs = self.blip.text_decoder.bert(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            return_dict=True,
            encoder_hidden_states=None,
        )
        text_embeds = cand_text_outputs.last_hidden_state[:, 0, :].unsqueeze(1)

        # Broadcast visual features across candidate beams (KV cache reuse principle)
        img_input_for_eval = image_embeds.expand(num_beams, -1, -1)
        enhanced_vis_check = self.adapter(img_input_for_eval, text_embeds)
        adapter_scores = torch.cosine_similarity(
            enhanced_vis_check, text_embeds, dim=-1
        ).squeeze(-1)

        # Min-Max normalization to align scale discrepancies between streams
        lang_norm = (lang_scores - lang_scores.min()) / (
            lang_scores.max() - lang_scores.min() + 1e-6
        )
        adapt_norm = (adapter_scores - adapter_scores.min()) / (
            adapter_scores.max() - adapter_scores.min() + 1e-6
        )

        final_scores = gamma * adapt_norm + (1.0 - gamma) * lang_norm
        best_idx = int(final_scores.argmax().item())

        candidate_records = []
        dpo_pair = None
        for idx in range(num_beams):
            rec = {
                "rank_index": idx,
                "caption": candidates[idx],
                "s_lang_norm": round(float(lang_norm[idx].item()), 4),
                "s_vis_norm": round(float(adapt_norm[idx].item()), 4),
                "s_vis_raw": round(float(adapter_scores[idx].item()), 4),
                "s_final": round(float(final_scores[idx].item()), 4),
            }
            candidate_records.append(rec)

            # Identify hard negative (high fluency score, low visual grounding score)
            if (
                dpo_pair is None
                and idx != best_idx
                and rec["s_lang_norm"] >= 0.75
                and rec["s_vis_norm"] <= 0.40
            ):
                dpo_pair = {
                    "chosen": candidates[best_idx],
                    "rejected": candidates[idx],
                    "chosen_s_vis": round(float(adapt_norm[best_idx].item()), 4),
                    "rejected_s_lang": rec["s_lang_norm"],
                    "rejected_s_vis": rec["s_vis_norm"],
                }
                if dpo_export_path:
                    with open(dpo_export_path, "a", encoding="utf-8") as dpo_file:
                        dpo_file.write(json.dumps(dpo_pair, ensure_ascii=False) + "\n")

        return {
            "best_caption": candidates[best_idx],
            "best_s_final": round(float(final_scores[best_idx].item()), 4),
            "best_s_vis_norm": round(float(adapt_norm[best_idx].item()), 4),
            "best_s_vis_raw": round(float(adapter_scores[best_idx].item()), 4),
            "best_s_lang_norm": round(float(lang_norm[best_idx].item()), 4),
            "salient_bbox": salient_bbox,
            "candidates": candidate_records,
            "dpo_pair": dpo_pair,
        }