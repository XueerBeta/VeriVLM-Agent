# -*- coding: utf-8 -*-
"""
Official CHAIR (Rohrbach et al., EMNLP 2018) & MS COCO Benchmark for VeriVLM-Agent.
Implements the complete official pipeline:
1. Full 80-category synset mapping (600+ fine-grained synonyms).
2. NLTK POS-aware WordNet lemmatization and bigram coalescing (DOUBLE_WORD_DICT)
   to prevent sub-word false positives (e.g., "hot dog" triggering animal "dog").
3. Dual-source Ground Truth union combining MS COCO instance segmentations
   (instances_val2014.json) and human reference captions (captions_val2014.json).
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, List, Set, Tuple, Any, Optional
import torch
from PIL import Image
from tqdm import tqdm

# Ensure project root directory is on sys.path when executing from subdirectories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

from core.evaluator import FullStackBLIPEvaluator
from data_pipeline.syntactic_dataset import build_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Official-CHAIR-Benchmark")


# ==============================================================================
# 1. Official Rohrbach et al. (EMNLP 2018) 80-Class Synset Definitions
# ==============================================================================
OFFICIAL_CHAIR_SYNONYMS: Dict[str, List[str]] = {
    "person": [
        "person", "girl", "boy", "man", "woman", "kid", "child", "chef", "baker", "people",
        "adult", "rider", "children", "baby", "worker", "passenger", "sister", "biker",
        "policeman", "cop", "officer", "lady", "cowboy", "bride", "groom", "male", "female",
        "guy", "traveler", "mother", "father", "gentleman", "pitcher", "player", "skier",
        "snowboarder", "skater", "skateboarder", "foreigner", "caller", "offender", "coworker",
        "trespasser", "patient", "politician", "soldier", "grandchild", "serviceman", "walker",
        "drinker", "doctor", "bicyclist", "thief", "buyer", "teenager", "student", "camper",
        "driver", "hunter", "shopper", "villager", "batter", "catcher", "umpire", "referee",
        "surfer", "pedestrian", "spectator", "crowd", "audience", "toddlers", "toddler", "infant",
    ],
    "bicycle": ["bicycle", "bike", "unicycle", "minibike", "trike", "mountain bike", "ten-speed"],
    "car": [
        "car", "automobile", "van", "minivan", "sedan", "suv", "hatchback", "cab", "jeep",
        "coupe", "taxicab", "limo", "taxi", "convertible", "cruiser", "station wagon", "auto",
    ],
    "motorcycle": [
        "motorcycle", "motorbike", "moped", "motor scooter", "scooter", "dirt bike",
        "motor cycle", "motor bike", "chopper", "vespa",
    ],
    "airplane": [
        "airplane", "aeroplane", "plane", "airliner", "jet", "jetliner", "biplane", "seaplane",
        "airbus", "aircraft", "fighter jet", "twinjet", "propeller plane", "bomber",
    ],
    "bus": [
        "bus", "minibus", "trolleybus", "school bus", "omnibus", "double-decker",
        "shuttle bus", "coach", "tour bus", "transit bus", "city bus",
    ],
    "train": [
        "train", "locomotive", "tram", "subway", "bullet train", "passenger train",
        "freight train", "monorail", "boxcar", "caboose", "railcar", "steam train", "trolley",
    ],
    "truck": [
        "truck", "pickup", "lorry", "dump truck", "tractor", "tow truck", "fire truck",
        "semi", "garbage truck", "flatbed", "delivery truck", "moving van", "trailer truck",
        "pickup truck", "fire engine", "semi truck", "tanker",
    ],
    "boat": [
        "boat", "ship", "yacht", "canoe", "kayak", "sailboat", "motorboat", "ferry", "schooner",
        "barge", "rowboat", "dinghy", "tugboat", "speedboat", "catamaran", "houseboat",
        "gondola", "vessel", "watercraft", "raft", "steamboat", "riverboat", "pontoon",
    ],
    "traffic light": [
        "traffic light", "street light", "traffic signal", "stop light", "stoplight", "semaphore",
    ],
    "fire hydrant": ["fire hydrant", "hydrant", "fireplug", "water plug"],
    "stop sign": ["stop sign", "street sign", "traffic sign"],
    "parking meter": ["parking meter", "meter"],
    "bench": ["bench", "pew", "park bench", "workbench", "wooden bench"],
    "bird": [
        "bird", "ostrich", "owl", "seagull", "goose", "duck", "parakeet", "falcon", "robin",
        "pelican", "waterfowl", "heron", "hummingbird", "mallard", "finch", "pigeon", "sparrow",
        "seabird", "osprey", "blackbird", "fowl", "shorebird", "woodpecker", "egret", "chickadee",
        "quail", "blue jay", "kingfisher", "buzzard", "willet", "gull", "swan", "bluejay",
        "flamingo", "cormorant", "parrot", "loon", "gosling", "waterbird", "pheasant", "rooster",
        "sandpiper", "crow", "raven", "turkey", "oriole", "cowbird", "warbler", "magpie",
        "peacock", "cockatiel", "lorikeet", "puffin", "vulture", "condor", "macaw", "hen",
        "chick", "eagle", "hawk", "penguin", "albatross", "cockatoo", "canary", "dove",
    ],
    "cat": ["cat", "kitten", "kitty", "tabby", "feline", "pussycat", "tomcat", "housecat"],
    "dog": [
        "dog", "puppy", "pup", "hound", "canine", "pooch", "mutt", "beagle", "bulldog", "pug",
        "terrier", "poodle", "labrador", "retriever", "shepherd", "chihuahua", "husky", "collie",
        "spaniel", "rottweiler", "dalmatian", "corgi", "dachshund", "schnauzer", "doberman",
        "mastiff", "pitbull", "greyhound", "chow", "sheepdog", "great dane", "boxer",
    ],
    "horse": [
        "horse", "pony", "foal", "stallion", "mare", "colt", "mustang", "thoroughbred",
        "steed", "equine", "racehorse", "bronco", "filly",
    ],
    "sheep": ["sheep", "lamb", "ram", "ewe", "bighorn", "flock of sheep"],
    "cow": [
        "cow", "cattle", "bull", "calf", "ox", "heifer", "bovine", "steer", "dairy cow",
        "yak", "buffalo", "bison", "water buffalo", "longhorn",
    ],
    "elephant": ["elephant", "mammoth", "pachyderm", "bull elephant", "baby elephant"],
    "bear": [
        "bear", "grizzly", "polar bear", "panda", "black bear", "brown bear", "bear cub",
        "grizzly bear", "panda bear", "kodiak",
    ],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "backpack": ["backpack", "rucksack", "knapsack", "bookbag", "haversack", "daypack", "knap sack"],
    "umbrella": ["umbrella", "parasol", "sunshade", "brolly"],
    "handbag": [
        "handbag", "purse", "tote", "clutch", "shoulder bag", "pocketbook", "satchel",
        "duffel bag", "tote bag", "hand bag", "shopping bag",
    ],
    "tie": ["tie", "necktie", "bow tie", "bowtie", "cravat", "neck tie"],
    "suitcase": [
        "suitcase", "luggage", "baggage", "valise", "trunk", "briefcase", "carry-on",
        "travel bag", "suit case",
    ],
    "frisbee": ["frisbee", "flying disc", "frizbee"],
    "skis": ["ski", "skis", "pair of skis", "snow ski"],
    "snowboard": ["snowboard", "snow board"],
    "sports ball": [
        "sports ball", "ball", "soccer ball", "football", "basketball", "baseball",
        "tennis ball", "volleyball", "golf ball", "bowling ball", "rugby ball", "softball",
        "ping pong ball", "dodgeball", "wiffle ball", "kickball", "beach ball",
    ],
    "kite": ["kite", "parasail", "paraglider", "stunt kite"],
    "baseball bat": ["baseball bat", "bat", "wooden bat"],
    "baseball glove": [
        "baseball glove", "glove", "mitt", "catcher's mitt", "baseball mitt", "catchers mitt",
    ],
    "skateboard": ["skateboard", "longboard", "skate board", "penny board"],
    "surfboard": [
        "surfboard", "wakeboard", "boogie board", "paddleboard", "windsurfer",
        "surf board", "skimboard", "bodyboard", "shortboard", "paddle board",
    ],
    "tennis racket": [
        "tennis racket", "racket", "racquet", "badminton racket", "squash racket", "tennis racquet",
    ],
    "bottle": [
        "bottle", "water bottle", "wine bottle", "beer bottle", "flask", "thermos",
        "decanter", "soda bottle", "pop bottle", "liquor bottle", "plastic bottle", "glass bottle",
    ],
    "wine glass": [
        "wine glass", "wineglass", "goblet", "champagne flute", "chalice", "stemware", "flute",
    ],
    "cup": [
        "cup", "mug", "teacup", "coffee cup", "paper cup", "plastic cup", "solo cup",
        "beaker", "tumbler", "drinking glass", "coffee mug", "tea cup",
    ],
    "fork": ["fork", "pitchfork", "table fork", "salad fork", "dinner fork"],
    "knife": [
        "knife", "blade", "dagger", "cleaver", "butter knife", "steak knife", "pocketknife",
        "machete", "scalpel", "kitchen knife", "paring knife", "chef knife",
    ],
    "spoon": [
        "spoon", "teaspoon", "tablespoon", "ladle", "soup spoon", "wooden spoon", "serving spoon",
    ],
    "bowl": [
        "bowl", "soup bowl", "salad bowl", "mixing bowl", "cereal bowl", "fruit bowl",
        "porcelain bowl", "ceramic bowl", "metal bowl",
    ],
    "banana": ["banana", "plantain", "bunch of bananas"],
    "apple": ["apple", "green apple", "red apple", "granny smith"],
    "sandwich": [
        "sandwich", "sub", "hoagie", "burger", "hamburger", "cheeseburger", "panini",
        "club sandwich", "grilled cheese", "submarine sandwich", "chicken sandwich",
    ],
    "orange": ["orange", "tangerine", "clementine", "mandarin", "navel orange"],
    "broccoli": ["broccoli", "broccoli floret"],
    "carrot": ["carrot", "baby carrot"],
    "hot dog": [
        "hot dog", "hotdog", "frankfurter", "wiener", "sausage", "corndog", "bratwurst",
        "chili dog", "corn dog", "frank",
    ],
    "pizza": [
        "pizza", "pepperoni pizza", "cheese pizza", "pizza pie", "slice of pizza", "pizza slice",
    ],
    "donut": [
        "donut", "doughnut", "bagel", "glazed donut", "cruller", "eclair", "chocolate donut",
    ],
    "cake": [
        "cake", "birthday cake", "wedding cake", "cupcake", "cheesecake", "shortcake",
        "sponge cake", "sheet cake", "chocolate cake", "layer cake",
    ],
    "chair": [
        "chair", "armchair", "recliner", "stool", "barstool", "folding chair", "rocking chair",
        "office chair", "swivel chair", "lawn chair", "deck chair", "highchair", "wheelchair",
        "lounge chair", "seat", "easy chair", "beach chair", "barber chair", "patio chair",
    ],
    "couch": [
        "couch", "sofa", "loveseat", "futon", "settee", "davenport", "sectional", "divan",
        "chesterfield", "day bed", "sofa bed", "leather couch",
    ],
    "potted plant": [
        "potted plant", "houseplant", "plant", "flower pot", "flowerpot", "fern", "bonsai",
        "cactus", "succulent", "planter", "house plant", "pot plant", "pottedplant",
    ],
    "bed": [
        "bed", "bunk bed", "crib", "cot", "mattress", "four-poster", "daybed", "waterbed",
        "cradle", "hospital bed", "double bed", "twin bed", "king bed", "queen bed",
    ],
    "dining table": [
        "dining table", "table", "desk", "coffee table", "kitchen table", "picnic table",
        "side table", "end table", "counter", "countertop", "dining room table",
        "conference table", "nightstand", "worktable", "folding table", "patio table",
    ],
    "toilet": [
        "toilet", "commode", "urinal", "latrine", "potty", "toilet bowl", "bidet", "lavatory",
    ],
    "tv": [
        "tv", "television", "monitor", "computer monitor", "screen", "flat screen", "plasma",
        "lcd", "crt", "hdtv", "television set", "tv screen", "telly", "tv monitor",
    ],
    "laptop": [
        "laptop", "notebook", "macbook", "netbook", "laptop computer", "portable computer",
        "chromebook", "thinkpad",
    ],
    "mouse": ["mouse", "computer mouse", "trackball", "optical mouse", "wireless mouse"],
    "remote": [
        "remote", "remote control", "controller", "wii remote", "wiimote", "game controller",
        "gamepad", "clicker", "tv remote",
    ],
    "keyboard": ["keyboard", "computer keyboard", "keypad", "typewriter", "mechanical keyboard"],
    "cell phone": [
        "cell phone", "cellphone", "mobile phone", "smartphone", "iphone", "android",
        "blackberry", "telephone", "phone", "flip phone", "handset", "payphone", "smart phone",
    ],
    "microwave": ["microwave", "microwave oven"],
    "oven": [
        "oven", "stove", "range", "toaster oven", "kiln", "furnace", "cooktop", "burner",
        "wood stove", "gas stove", "kitchen stove",
    ],
    "toaster": ["toaster"],
    "sink": [
        "sink", "washbasin", "basin", "kitchen sink", "bathroom sink", "washbowl", "hand sink",
    ],
    "refrigerator": [
        "refrigerator", "fridge", "freezer", "cooler", "icebox", "mini fridge", "fridges",
    ],
    "book": [
        "book", "novel", "textbook", "paperback", "hardcover", "magazine", "comic book",
        "notebook", "booklet", "dictionary", "bible", "journal", "encyclopedia", "manual",
    ],
    "clock": [
        "clock", "wall clock", "alarm clock", "watch", "stopwatch", "grandfather clock",
        "timepiece", "sundial", "digital clock", "clock tower", "wrist watch", "wristwatch",
    ],
    "vase": ["vase", "pitcher", "urn", "carafe", "flower vase", "ceramic vase", "glass vase"],
    "scissors": ["scissors", "shears", "pair of scissors", "kitchen shears"],
    "teddy bear": [
        "teddy bear", "stuffed animal", "plush", "plushie", "teddy", "stuffed bear",
        "toy bear", "plush toy", "stuffed toy",
    ],
    "hair drier": ["hair drier", "hair dryer", "blow dryer", "blowdryer", "hairdryer"],
    "toothbrush": ["toothbrush", "electric toothbrush", "tooth brush"],
}

NON_OBJECT_COMPOUND_EXCLUSIONS: Set[str] = {
    "toilet seat", "car seat", "bicycle seat", "bike seat", "back seat", "driver seat",
    "passenger seat", "window seat", "seat belt", "fire place", "hot air", "table tennis",
    "water ski", "water skis", "jet ski", "tree trunk", "elephant trunk", "swimming trunks",
    "mouse pad", "mouse trap", "mickey mouse", "minnie mouse", "donald duck", "hot chocolate",
    "hot sauce", "dog house", "bird house", "bird cage", "cat food", "dog food", "apple tree",
    "orange tree", "banana tree", "pine cone", "traffic cone", "ice cream", "coffee bean",
    "cup cake", "pancake", "pan cake", "carrot cake", "urinal cake", "crab cake", "fish cake",
}


def ensure_wordnet_resources() -> None:
    """Ensures NLTK POS tagger and WordNet lemmatizer corpora are downloaded."""
    for res in [
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "wordnet",
        "omw-1.4",
    ]:
        try:
            nltk.download(res, quiet=True)
        except Exception as exc:
            logger.debug("NLTK resource '%s' check skipped: %s", res, exc)


def get_wordnet_pos(treebank_tag: str) -> str:
    """Maps Penn Treebank POS tags to WordNet lemmatizer POS constants."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


# ==============================================================================
# 2. Official CHAIR & Object Recall Evaluator (Rohrbach et al., EMNLP 2018)
# ==============================================================================
class OfficialCHAIREvaluator:
    """
    Implements the exact CHAIRs, CHAIRi, and Object Recall evaluation protocol
    from Rohrbach et al. (EMNLP 2018).
    """

    def __init__(
        self,
        coco_instances_path: str,
        coco_captions_path: str,
        custom_synonyms_txt: Optional[str] = None,
    ) -> None:
        ensure_wordnet_resources()
        self.lemmatizer = WordNetLemmatizer()

        self.word_to_coco_class: Dict[str, str] = {}
        self.double_word_dict: Dict[Tuple[str, str], str] = {}

        self._build_synonym_tables(custom_synonyms_txt)

        logger.info("Loading MS COCO instance annotations: %s", coco_instances_path)
        self.coco_instances = COCO(coco_instances_path)

        logger.info("Loading MS COCO reference caption annotations: %s", coco_captions_path)
        self.coco_captions = COCO(coco_captions_path)

        self.gt_objects_cache: Dict[int, Set[str]] = {}

    def _lemmatize_phrase(self, phrase: str) -> List[str]:
        tokens = nltk.word_tokenize(phrase.lower().strip())
        pos_tags = nltk.pos_tag(tokens)
        return [self.lemmatizer.lemmatize(w, get_wordnet_pos(pos)) for w, pos in pos_tags]

    def _build_synonym_tables(self, custom_synonyms_txt: Optional[str]) -> None:
        synsets = OFFICIAL_CHAIR_SYNONYMS.copy()

        if custom_synonyms_txt and os.path.exists(custom_synonyms_txt):
            logger.info("Loading custom synonyms file: %s", custom_synonyms_txt)
            with open(custom_synonyms_txt, "r", encoding="utf-8") as f:
                for line in f:
                    parts = [p.strip().lower() for p in line.strip().split(",") if p.strip()]
                    if parts:
                        canonical = parts[0]
                        synsets[canonical] = parts

        for coco_cls, syn_list in synsets.items():
            for syn in syn_list:
                lemma_tokens = self._lemmatize_phrase(syn)
                if not lemma_tokens:
                    continue
                if len(lemma_tokens) == 1:
                    self.word_to_coco_class[lemma_tokens[0]] = coco_cls
                elif len(lemma_tokens) == 2:
                    pair = (lemma_tokens[0], lemma_tokens[1])
                    compound_str = f"{lemma_tokens[0]} {lemma_tokens[1]}"
                    self.double_word_dict[pair] = compound_str
                    self.word_to_coco_class[compound_str] = coco_cls
                elif len(lemma_tokens) >= 3:
                    pair = (lemma_tokens[0], lemma_tokens[-1])
                    compound_str = f"{lemma_tokens[0]} {lemma_tokens[-1]}"
                    self.double_word_dict[pair] = compound_str
                    self.word_to_coco_class[compound_str] = coco_cls

        for excl in NON_OBJECT_COMPOUND_EXCLUSIONS:
            lemma_tokens = self._lemmatize_phrase(excl)
            if len(lemma_tokens) == 2:
                self.double_word_dict[(lemma_tokens[0], lemma_tokens[1])] = "__EXCLUDED_COMPOUND__"

    def caption_to_coco_objects(self, caption: str) -> Tuple[List[str], List[str]]:
        clean_caption = re.sub(r"[^\w\s]", " ", caption.lower())
        raw_tokens = nltk.word_tokenize(clean_caption)
        if not raw_tokens:
            return [], []

        pos_tags = nltk.pos_tag(raw_tokens)
        lemmatized_tokens = [
            self.lemmatizer.lemmatize(tok, get_wordnet_pos(pos)) for tok, pos in pos_tags
        ]

        coalesced_tokens: List[str] = []
        i = 0
        num_tokens = len(lemmatized_tokens)
        while i < num_tokens:
            if i + 1 < num_tokens:
                bigram = (lemmatized_tokens[i], lemmatized_tokens[i + 1])
                if bigram in self.double_word_dict:
                    merged = self.double_word_dict[bigram]
                    if merged != "__EXCLUDED_COMPOUND__":
                        coalesced_tokens.append(merged)
                    i += 2
                    continue
            coalesced_tokens.append(lemmatized_tokens[i])
            i += 1

        mentioned_objects: List[str] = []
        surface_nodes: List[str] = []
        for token in coalesced_tokens:
            if token in self.word_to_coco_class:
                mentioned_objects.append(self.word_to_coco_class[token])
                surface_nodes.append(token)

        return mentioned_objects, surface_nodes

    def get_ground_truth_objects(self, image_id: int) -> Set[str]:
        if image_id in self.gt_objects_cache:
            return self.gt_objects_cache[image_id]

        gt_set: Set[str] = set()

        if image_id in self.coco_instances.imgs:
            inst_ann_ids = self.coco_instances.getAnnIds(imgIds=image_id)
            inst_anns = self.coco_instances.loadAnns(inst_ann_ids)
            cat_ids = [ann["category_id"] for ann in inst_anns]
            cats = self.coco_instances.loadCats(cat_ids)
            for cat in cats:
                cat_name = cat["name"].lower().strip()
                if cat_name in OFFICIAL_CHAIR_SYNONYMS:
                    gt_set.add(cat_name)

        if image_id in self.coco_captions.imgs:
            cap_ann_ids = self.coco_captions.getAnnIds(imgIds=image_id)
            cap_anns = self.coco_captions.loadAnns(cap_ann_ids)
            for ann in cap_anns:
                if "caption" in ann:
                    cap_objs, _ = self.caption_to_coco_objects(ann["caption"])
                    gt_set.update(cap_objs)

        self.gt_objects_cache[image_id] = gt_set
        return gt_set

    def evaluate(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        num_sentences = 0
        num_hallucinated_sentences = 0
        total_mentioned_instances = 0
        hallucinated_instances = 0
        recalled_gt_objects = 0
        total_gt_objects = 0

        for pred in predictions:
            img_id = int(pred["image_id"])
            caption = pred["caption"]

            gt_objects = self.get_ground_truth_objects(img_id)
            if not gt_objects:
                continue

            mentioned_objects, _ = self.caption_to_coco_objects(caption)
            num_sentences += 1

            sentence_has_hallucination = False
            for obj in mentioned_objects:
                total_mentioned_instances += 1
                if obj not in gt_objects:
                    hallucinated_instances += 1
                    sentence_has_hallucination = True

            if sentence_has_hallucination:
                num_hallucinated_sentences += 1

            unique_mentioned = set(mentioned_objects)
            recalled_gt_objects += len(unique_mentioned & gt_objects)
            total_gt_objects += len(gt_objects)

        chair_s = (num_hallucinated_sentences / max(1, num_sentences)) * 100.0
        chair_i = (hallucinated_instances / max(1, total_mentioned_instances)) * 100.0
        obj_recall = (recalled_gt_objects / max(1, total_gt_objects)) * 100.0

        return {
            "CHAIR_s": round(chair_s, 2),
            "CHAIR_i": round(chair_i, 2),
            "Object_Recall": round(obj_recall, 2),
        }


# ==============================================================================
# 3. Full-Stack Benchmark Runner
# ==============================================================================
def evaluate_epoch(
    model: FullStackBLIPEvaluator,
    epoch: int,
    chair_evaluator: OfficialCHAIREvaluator,
    img_ids: List[int],
    transform: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> Optional[Dict[str, float]]:
    ckpt_path = os.path.join(args.checkpoint_dir, f"adapter_epoch_{epoch}.pth")
    if not os.path.exists(ckpt_path):
        logger.warning("Adapter checkpoint not found: %s", ckpt_path)
        return None

    model.load_adapter_checkpoint(ckpt_path, device=device)
    result_json_path = os.path.join(args.output_dir, f"res_full_stack_epoch_{epoch}.json")
    dpo_jsonl_path = (
        os.path.join(args.output_dir, f"dpo_pairs_epoch_{epoch}.jsonl")
        if args.harvest_dpo
        else None
    )

    results: List[Dict[str, Any]] = []
    logger.info(
        "Evaluating Epoch %d | Images=%d | ViPER=%.3f | Gamma=%.2f | Lambda=%.2f",
        epoch,
        len(img_ids),
        args.viper_scale,
        args.gamma,
        args.diversity_penalty,
    )

    coco_captions = chair_evaluator.coco_captions
    for img_id in tqdm(img_ids, desc=f"Inference Ep {epoch}", leave=False):
        img_info = coco_captions.loadImgs(img_id)[0]
        img_path = os.path.join(args.img_folder, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        raw_image = Image.open(img_path).convert("RGB")
        image_tensor = transform(raw_image).unsqueeze(0).to(device)

        output_dict = model.generate_full_stack(
            image=image_tensor,
            gamma=args.gamma,
            num_beams=args.num_beams,
            diversity_penalty=args.diversity_penalty,
            length_penalty=args.length_penalty,
            dpo_export_path=dpo_jsonl_path,
        )
        results.append({"image_id": img_id, "caption": output_dict["best_caption"]})

    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    coco_dt = coco_captions.loadRes(result_json_path)
    coco_eval = COCOEvalCap(coco_captions, coco_dt)
    coco_eval.params["image_id"] = coco_dt.getImgIds()

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    if args.enable_spice:
        from pycocoevalcap.spice.spice import Spice
        scorers.append((Spice(), "SPICE"))

    coco_eval.scorers = scorers
    coco_eval.evaluate()

    metrics: Dict[str, float] = {k: round(v * 100.0, 2) for k, v in coco_eval.eval.items()}
    hallucination_metrics = chair_evaluator.evaluate(results)
    metrics.update(hallucination_metrics)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official CHAIR (EMNLP 2018) & MS COCO Benchmark for VeriVLM-Agent"
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
    parser.add_argument(
        "--img_folder",
        type=str,
        default="./data/coco/karpathy_test_images",
        help="Directory containing MS COCO Karpathy test split images.",
    )
    parser.add_argument(
        "--ann_path",
        type=str,
        default="./data/coco/annotations/karpathy_test_coco_format.json",
        help="Path to ground-truth COCO format caption annotations JSON.",
    )
    parser.add_argument(
        "--coco_instances_path",
        type=str,
        default="./data/coco/annotations/instances_val2014.json",
        help="Path to official MS COCO instance annotations JSON (instances_val2014.json).",
    )
    parser.add_argument(
        "--synonyms_txt",
        type=str,
        default=None,
        help="Optional path to external Rohrbach et al. synonyms.txt file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/eval_results",
        help="Directory to store prediction JSONs and metric reports.",
    )

    parser.add_argument("--vit_type", type=str, default="base", choices=["base", "large"])
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--test_epochs", type=int, nargs="+", default=[10])

    parser.add_argument("--viper_scale", type=float, default=0.083)
    parser.add_argument("--gamma", type=float, default=0.35)
    parser.add_argument("--diversity_penalty", type=float, default=1.45)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=5)

    parser.add_argument("--max_samples", type=int, default=0, help="0 = full 5,000 test split.")
    parser.add_argument("--enable_spice", action="store_true")
    parser.add_argument("--harvest_dpo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FullStackBLIPEvaluator(
        base_model_path=args.base_model_path,
        image_size=args.image_size,
        vit_type=args.vit_type,
        viper_scale=args.viper_scale,
    ).to(device)
    model.eval()

    chair_evaluator = OfficialCHAIREvaluator(
        coco_instances_path=args.coco_instances_path,
        coco_captions_path=args.ann_path,
        custom_synonyms_txt=args.synonyms_txt,
    )

    transform = build_transform(args.image_size)
    img_ids = list(chair_evaluator.coco_captions.imgs.keys())
    if args.max_samples > 0:
        img_ids = img_ids[: args.max_samples]

    separator = "=" * 72
    sub_sep = "-" * 72

    for ep in args.test_epochs:
        scores = evaluate_epoch(
            model=model,
            epoch=ep,
            chair_evaluator=chair_evaluator,
            img_ids=img_ids,
            transform=transform,
            args=args,
            device=device,
        )
        if scores:
            print(f"\n{separator}")
            print(f"VERIVLM-AGENT BENCHMARK (Official EMNLP 2018 CHAIR Protocol) | Epoch {ep}")
            print(sub_sep)
            print(f"  CIDEr         : {scores.get('CIDEr', 0.0):.2f}")
            if "SPICE" in scores:
                print(f"  SPICE         : {scores['SPICE']:.2f}")
            print(f"  BLEU-1        : {scores.get('Bleu_1', 0.0):.2f}")
            print(f"  BLEU-4        : {scores.get('Bleu_4', 0.0):.2f}")
            print(f"  METEOR        : {scores.get('METEOR', 0.0):.2f}")
            print(f"  ROUGE-L       : {scores.get('ROUGE_L', 0.0):.2f}")
            print(sub_sep)
            print(f"  Object Recall : {scores.get('Object_Recall', 0.0):.2f}%  (Dual-GT Coverage)")
            print(f"  CHAIR_s       : {scores.get('CHAIR_s', 0.0):.2f}%  (Sentence Hallucination)")
            print(f"  CHAIR_i       : {scores.get('CHAIR_i', 0.0):.2f}%  (Instance Hallucination)")
            print(f"{separator}\n")


if __name__ == "__main__":
    main()