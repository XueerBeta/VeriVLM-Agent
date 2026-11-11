# -*- coding: utf-8 -*-
"""
Data preprocessing and PyTorch Dataset pipeline for VeriVLM-Agent (Beta-BLIP).
Supports both offline syntactic noun-phrase (NP) extraction into JSON and
high-throughput multi-granular batch collation for Beta-SEA training.
"""

import json
import logging
import os
import random
from typing import List, Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import BertTokenizer
import nltk
from nltk.chunk import RegexpParser

logger = logging.getLogger(__name__)


def ensure_nltk_resources() -> None:
    """Downloads required NLTK tokenizers and POS taggers if missing."""
    resources = [
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ]
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception as exc:
            logger.debug("Optional NLTK resource '%s' skipped: %s", res, exc)


def init_blip_tokenizer(vocab_source: str = "bert-base-uncased") -> BertTokenizer:
    """Initializes BertTokenizer with BLIP-specific [DEC] and [ENC] control tokens."""
    tokenizer = BertTokenizer.from_pretrained(vocab_source)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    return tokenizer


def build_transform(image_size: int = 384) -> transforms.Compose:
    """Standard bicubic resize and ImageNet/CLIP normalization pipeline aligned with BLIP."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def extract_noun_phrases(caption: str) -> List[str]:
    """
    Extracts fine-grained noun phrases (NP) from a sentence via syntactic chunking.

    Args:
        caption: Input ground-truth description string.

    Returns:
        List of extracted noun phrase strings.
    """
    try:
        tokens = nltk.word_tokenize(caption)
        pos_tags = nltk.pos_tag(tokens)
        grammar = r"NP: {<DT>?<JJ.*>*<NN.*>+}"
        chunk_parser = RegexpParser(grammar)
        tree = chunk_parser.parse(pos_tags)
    except Exception as exc:
        logger.warning("Syntactic chunking failed for caption '%s': %s", caption, exc)
        return []

    phrases: List[str] = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
        phrase = " ".join(word for word, _ in subtree.leaves()).strip()
        if phrase:
            phrases.append(phrase)
    return phrases


class BetaCOCODataset(Dataset):
    """
    MS COCO dataset loader for multi-granular semantic alignment.
    Supports pre-extracted phrase JSON files (coco_karpathy_train_phrases.json)
    with automatic fallback to on-the-fly NLTK syntactic extraction.
    """

    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        max_phrases: int = 4,
        enable_online_chunking_fallback: bool = True,
        max_samples: int = 0,
    ) -> None:
        super().__init__()
        self.img_folder = img_folder
        self.max_phrases = max_phrases
        self.enable_online_chunking_fallback = enable_online_chunking_fallback

        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        logger.info("Loading multi-granular annotations from: %s", ann_file)
        with open(ann_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if isinstance(raw_data, list):
            annotations = raw_data
            self.id_to_filename = {}
        elif isinstance(raw_data, dict) and "annotations" in raw_data:
            self.id_to_filename = {
                img["id"]: img["file_name"] for img in raw_data.get("images", [])
            }
            annotations = raw_data["annotations"]
        else:
            raise ValueError("Unsupported annotation JSON format.")

        if max_samples > 0:
            annotations = annotations[:max_samples]
        self.annotations = annotations
        logger.info("Successfully loaded %d annotation samples.", len(self.annotations))

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]
        img_filename = ann.get("image") or ann.get("file_name")
        if img_filename is None and "image_id" in ann:
            img_filename = self.id_to_filename.get(ann["image_id"])

        full_path = os.path.join(self.img_folder, img_filename)
        caption = ann["caption"].strip()
        phrases = ann.get("phrases")

        if phrases is None:
            phrases = extract_noun_phrases(caption) if self.enable_online_chunking_fallback else []

        if len(phrases) > self.max_phrases:
            selected_phrases = random.sample(phrases, self.max_phrases)
        else:
            selected_phrases = list(phrases)

        text_list = [caption] + selected_phrases
        return {"img_path": full_path, "text_list": text_list}


class BlipCollate:
    """
    Collates image tensors and variable-length hierarchical text queries
    ([global_caption, phrase_1, phrase_2, ...]) along with sample-to-query mapping counts.
    """

    def __init__(self, tokenizer: Any, transform: Any, max_length: int = 35) -> None:
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        images: List[torch.Tensor] = []
        flat_texts: List[str] = []
        text_mapping: List[int] = []

        for item in batch:
            try:
                raw_image = Image.open(item["img_path"]).convert("RGB")
                image_tensor = self.transform(raw_image)
                images.append(image_tensor)

                flat_texts.extend(item["text_list"])
                text_mapping.append(len(item["text_list"]))
            except Exception as exc:
                logger.warning(
                    "Skipped corrupted or missing image '%s': %s", item.get("img_path"), exc
                )
                continue

        if not images:
            return None

        stacked_images = torch.stack(images, dim=0)
        text_inputs = self.tokenizer(
            flat_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "images": stacked_images,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "text_mapping": text_mapping,
        }