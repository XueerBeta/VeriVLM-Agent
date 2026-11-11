VeriVLM-Agent ($\beta$-BLIP)Closed-Loop Semantic Calibration, Self-Reflective Visual Agent & Hardware Roofline Benchmark for Vision-Language Models1. What is VeriVLM-Agent? (Executive Summary)VeriVLM-Agent is an open-source multimodal post-training, hallucination-governance, and self-reflective agent framework built upon our research paper ("$\beta$-BLIP: Closed-Loop Semantic Calibration for Generalizable and Lightweight Image Captioning").Standard Vision-Language Models (VLMs) suffer from the autoregressive likelihood trap: they overfit high-frequency generic phrases, lose fine-grained long-tail visual details, and hallucinate non-existent objects. VeriVLM-Agent solves this without scaling to massive, slow 7B+ decoders by upgrading the traditional unidirectional pipeline into a Perception $\rightarrow$ Alignment $\rightarrow$ Verification $\rightarrow$ Agentic Self-Reflection closed loop:Zero-Operator Visual Context Purification (ViPER): Uses token-level channel statistical variance and a learnable residual gate ($\alpha: 0.02 \rightarrow 0.083$) to amplify foreground targets and suppress background noise, boosting Object Recall from 44.14% to 53.26% with zero extra convolution or attention layers.Lightweight Multi-Granular Soft-Label Alignment (Beta-SEA): Extracts hierarchical "global sentence + local noun phrase" training pairs via an automated NLTK syntactic chunking pipeline, and trains a 5M-parameter (+2.2% capacity) cross-attention adapter with an asymmetric soft-label loss ($\beta = 0.5$) to eliminate cross-attention context bleeding. Single-module tuning improves MS COCO validation CIDEr by +6.8 points and cuts instance hallucination ($\text{CHAIR}_i$) from 15.46% to 8.76%.Zero-Extra-Parameter Visual Critic & DPO Data Flywheel (Gamma-DHR): Combines Grouped Beam Search ($\lambda = 1.45$, boosting long-tail concept frequency by $2\times\text{--}8\times$) with a dual-stream reranker ($\gamma = 0.35$) that reuses the 5M Beta-SEA adapter as an internal judge (MLLM-as-a-Judge). Automatically harvests "high fluency, low visual alignment" hard negatives to construct standardized DPO preference datasets.LangGraph ReAct Visual Self-Reflection Agent: Wraps the inference engine into a stateful cyclical graph. When posterior visual confidence falls below the safety guardrail under extreme degradation (micro-targets or heavy occlusion), the agent automatically triggers Function Calling to crop and bicubic-upsample high-variance regions for secondary verification.SOTA Accuracy with Edge-Grade Efficiency: Achieves 152.3 CIDEr (+14.6%), 24.8 SPICE, 58.40% Object Recall, and 7.96% $\text{CHAIR}_i$ on MS COCO (outperforming the 3.85B-parameter BLIP-2 by +6.5 CIDEr points), while requiring only 229M total parameters, ~2.90 GB peak VRAM, and ~95 ms latency (-84.5% FLOPs and $3.8\times$ speedup vs. TinyLLaVA-1.5B).2. End-to-End Closed-Loop Pipeline (Step-by-Step Dataflow)When an image is fed into VeriVLM-Agent, it flows through five sequential stages:[Input Image]
   |
   +---> Step 1: Visual Perception & Salience Localization (core/modules.py -> ViPER)
   |       * Frozen ViT-B/16 extracts raw visual patch embeddings V.
   |       * ViPER computes channel-wise variance sigma_i^2 for each patch token, applies global
   |         standardization and tanh routing, and modulates features via V_tilde = V * (1 + sigma(alpha) * W_perc).
   |       * Simultaneously extracts normalized bounding coordinates (x1, y1, x2, y2) of the highest-variance
   |         foreground region without any external object detector.
   |
   +---> Step 2: Pre-Decoding Task Prior Injection (core/evaluator.py -> Beta-SEA Feed-Forward)
   |       * The 5M Beta-SEA cross-attention adapter projects generic task prompt features onto V_tilde.
   |       * Injects the resulting semantic residual into the global [CLS] token (v_cls_tilde) to reduce
   |         initial decoding entropy before generating the first word.
   |
   +---> Step 3: Diversity-Guided Hypothesis Exploration (core/evaluator.py -> Grouped Beam Search)
   |       * Frozen BERT decoder runs Grouped Beam Search (B = 5 beams divided into G = 5 groups).
   |       * Applies inter-group repulsion penalty (lambda = 1.45) to prevent path-locking on generic templates,
   |         producing 5 diverse candidate captions C = {c_1, c_2, ..., c_5}.
   |
   +---> Step 4: Dual-Stream Posterior Verification (core/evaluator.py -> Gamma-DHR Visual Critic)
   |       * Stream 1 (Language Fluency S_lang): Length-normalized autoregressive log-likelihood.
   |       * Stream 2 (Visual Fidelity S_vis)  : Reuses the 5M Beta-SEA adapter as a zero-extra-parameter
   |         Visual Critic, feeding each candidate c_i back as a query to compute cosine alignment with V_tilde.
   |       * Min-Max normalizes both streams and computes S_final = 0.35 * N(S_vis) + 0.65 * N(S_lang).
   |
   +---> Step 5: Dual Closed-Loop Actions (Online Agent Reflection & Offline DPO Harvesting)
           |
           |-- [Action A: Online LangGraph Self-Reflective Agent (agent/verivlm_graph.py)]
           |     * Guardrail checks whether raw visual alignment S_vis_raw >= threshold (0.25).
           |     * If PASS: Immediately outputs the optimal verified caption.
           |     * If FAIL (and retry_count < max_retries): Triggers Function Calling to crop and 2x bicubic-zoom
           |       the high-variance bounding box from Step 1, routing back to Step 1 for secondary re-verification.
           |
           |-- [Action B: Automated DPO Preference Data Flywheel (outputs/*.jsonl)]
                 * Scans non-selected candidates for "Likelihood Trap" Hard Negatives (S_lang >= 0.75 & S_vis <= 0.40).
                 * Pairs the verified optimal caption (Chosen) with the hallucinated candidate (Rejected) and exports
                   standardized JSONL records for downstream Direct Preference Optimization (DPO).
3. Core Engineering & Algorithm Modulescore/modules.py (ViPER & BetaSemanticAdapter):Implements ViPER (Eq. 1-3) with inverse-Sigmoid warm-start initialization ($\alpha: 0.02 \rightarrow 0.083$) and detector-free high-density bounding box extraction (extract_high_density_bbox).Implements BetaSemanticAdapter (Eq. 4-6), a 5M-parameter multi-head cross-attention residual bottleneck shared between forward prior injection and posterior visual verification.core/evaluator.py (FullStackBLIPEvaluator):Integrates visual encoding, ViPER calibration, Beta-SEA prior injection, Grouped Beam Search, Gamma-DHR dual-stream reranking (Eq. 9-13), cross-beam visual feature broadcasting (KV cache reuse principle), and automated DPO pair harvesting.agent/verivlm_graph.py (VeriVLMAgentWorkflow):Implements the LangGraph cyclical state machine (AgentState) with conditional guardrail routing (should_trigger_reflection) and the localized crop-and-upsample tool node (adaptive_crop_zoom_tool_node).data_pipeline/syntactic_dataset.py (BetaCOCODataset & BlipCollate):Implements the NLTK syntactic chunking pipeline (NP: {<DT>?<JJ.*>*<NN.*>+}) that decomposes captions into global sentences and local entity/attribute phrases.train_beta_sea.py & train_stage3_viper_scst.py (Progressive 3-Stage Training):Follows the "Align First, Decode Second, Perceive Last" optimization strategy:train_beta_sea.py: Freezes the 224M BLIP backbone and optimizes Beta-SEA under the block-diagonal asymmetric soft-label objective (BetaCALLoss, $\beta = 0.5$, $\tau = 0.07$).train_stage3_viper_scst.py: Jointly optimizes the ViPER scalar ($\text{lr} = 1\times 10^{-4}$) and Beta-SEA adapter ($\text{lr} = 2\times 10^{-4}$) using Automatic Mixed Precision (AMP), 4-step gradient accumulation (effective batch size 64), and Self-Critical Sequence Training (SCST) CIDEr reinforcement learning.benchmark_eval/ (Official CHAIR & 14-MLLM Hardware Roofline Suite):evaluate_coco_and_chair.py: Computes standard MS COCO metrics (BLEU, METEOR, ROUGE-L, CIDEr, SPICE) alongside the official EMNLP 2018 CHAIR protocol (600+ synset dictionary, POS-aware WordNet lemmatization, bigram coalescing to prevent sub-word false positives, and dual-source Ground Truth union of instances_val2014.json + reference captions).roofline_profiler.py: Profiles parameter scale, peak VRAM, inference latency, GFLOPs, and arithmetic intensity across 14 mainstream MLLMs (including LLaVA-1.5, InstructBLIP, BLIP-2, PaliGemma, MiniCPM-V, TinyLLaVA, FastVLM, SmolVLM, and $\beta$-BLIP) under a unified FP16 benchmark on an 8 GB NVIDIA RTX 4060 GPU.visualization/plot_attention_heatmap.py (FullStackAttentionVisualizer):Hooks into Beta-SEA multi-head cross-attention layers and computes Strict Contrastive Attention (target word attention minus sentence-average background attention) to render fine-grained visual grounding heatmaps.4. Repository Structure & Pre-Generated Outputs (./outputs/)All visual charts, attention grounding heatmaps, 14-MLLM Hardware Roofline comparison tables (.csv / .json), and harvested DPO preference datasets are stored directly inside the ./outputs/ directory:VeriVLM-Agent/
├── README.md                               # Project documentation
├── requirements.txt                        # Pinned Python 3.9+ / CUDA 12.1 dependencies
├── run_inference.py                        # CLI entry for single-pass & LangGraph Agent inference
├── train_beta_sea.py                       # Stage-1: Beta-SEA multi-granular soft-label alignment
├── train_stage3_viper_scst.py              # Stage-3: Joint ViPER (alpha) & CIDEr SCST optimization
├── models/                                 # BLIP backbone definitions (vit.py, med.py, blip.py)
├── core/
│   ├── modules.py                          # ViPER (Eq. 1-3) & BetaSemanticAdapter (Eq. 4-6)
│   └── evaluator.py                        # FullStackBLIPEvaluator, Gamma-DHR & DPO harvester
├── agent/
│   └── verivlm_graph.py                    # LangGraph ReAct state machine & crop-zoom tool
├── data_pipeline/
│   └── syntactic_dataset.py                # NLTK syntactic NP chunker & multi-granular collator
├── benchmark_eval/
│   ├── evaluate_coco_and_chair.py          # MS COCO metrics + official EMNLP 2018 CHAIR evaluator
│   └── roofline_profiler.py                # 14-model MLLM Hardware Roofline efficiency profiler
├── visualization/
│   └── plot_attention_heatmap.py           # Strict contrastive cross-attention heatmap generator
└── outputs/                                # Generated charts, hardware benchmark tables & DPO logs
    ├── roofline_benchmark/                 # 14-MLLM efficiency comparison reports (.csv & .json)
    ├── heatmaps/                           # Contrastive cross-attention grounding heatmaps (.png)
    └── eval_results/                       # Evaluation logs & harvested DPO preference pairs
5. Installation# 1. Clone the repository
git clone https://github.com/XueerBeta/VeriVLM-Agent.git
cd VeriVLM-Agent

# 2. Create and activate Conda environment (Python 3.9 + CUDA 12.1 recommended)
conda create -n verivlm python=3.9 -y
conda activate verivlm

# 3. Install pinned dependencies
pip install -r requirements.txt
6. Quick Start & Reproduction Commands6.1 Single-Image Inference & LangGraph Self-Reflective Agent# Run in LangGraph ReAct Agent mode (automatically triggers crop-zoom re-verification if S_vis < 0.25)
python run_inference.py \
  --image_path ./assets/demo_example.jpg \
  --mode agent \
  --viper_scale 0.083 \
  --gamma 0.35 \
  --diversity_penalty 1.45 \
  --vis_threshold 0.25 \
  --dpo_export_path ./outputs/dpo_harvested_pairs.jsonl

# Run in standard single-pass closed-loop mode
python run_inference.py --image_path ./assets/demo_example.jpg --mode standard
6.2 Progressive Three-Stage Training# Stage 1: Train 5M Beta-SEA Adapter with Asymmetric Soft-Label Loss (beta = 0.5)
python train_beta_sea.py \
  --base_model_path ./pretrained/model_base_caption_capfilt_large.pth \
  --img_root ./data/coco/train2014 \
  --json_path ./data/coco/annotations/coco_karpathy_train_phrases.json \
  --save_dir ./checkpoints/checkpoints_beta \
  --epochs 10 --batch_size 2 --accum_steps 16 --lr 3e-4 --beta 0.5

# Stage 3: Jointly Optimize ViPER Scalar (alpha: 0.02 -> 0.083) & Beta-SEA via CIDEr SCST
python train_stage3_viper_scst.py \
  --base_model_path ./pretrained/model_base_caption_capfilt_large.pth \
  --stage1_adapter_ckpt ./checkpoints/checkpoints_beta/adapter_epoch_10.pth \
  --img_root ./data/coco/train2014 \
  --ann_json_path ./data/coco/annotations/captions_train2014.json \
  --output_dir ./checkpoints/checkpoints_stage3_viper \
  --mode scst --batch_size 16 --grad_accum_steps 4 --lr_viper 1e-4 --lr_adapter 2e-4
6.3 Generate 14-MLLM Hardware Roofline Benchmark & Visualizations (Saved to ./outputs/)# 1. Generate 14-MLLM Hardware Roofline Efficiency Table (exports CSV/JSON to ./outputs/roofline_benchmark/)
python benchmark_eval/roofline_profiler.py --run_live_probe

# 2. Run MS COCO & Official EMNLP 2018 CHAIR Evaluation (exports to ./outputs/eval_results/)
python benchmark_eval/evaluate_coco_and_chair.py \
  --viper_scale 0.083 --gamma 0.35 --diversity_penalty 1.45 --harvest_dpo

# 3. Generate Contrastive Attention Heatmaps (exports PNG to ./outputs/heatmaps/)
python visualization/plot_attention_heatmap.py \
  --image_path ./assets/demo_example.jpg \
  --phrase_query "a baby" \
  --target_word "baby" \
  --output_dir ./outputs/heatmaps
7. Citation & Acknowledgements@article{xu2026betablip,
  title   = {$\beta$-BLIP: Closed-Loop Semantic Calibration for Generalizable and Lightweight Image Captioning},
  author  = {Xu, Zhigang and Wang, Shen and Dong, Xinhua and Han, Hongmu and Zhang, Xingyu and Wang, Wenbo},
  journal = {Neurocomputing},
  year    = {2026},
  note    = {Under Review}
}
Acknowledgements: Supported by the Science and Technology Projects of Xizang Autonomous Region, China (Grant No. XZ202601ZYDX005), and the Innovation Fund of Hubei University of Technology (Grant Nos. BSQD2019027, BSQD2019020, and BSQD2016019).