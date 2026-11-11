# -*- coding: utf-8 -*-
"""
LangGraph-based ReAct visual self-reflection workflow for VeriVLM-Agent.
Triggers localized high-density region cropping and re-verification when
posterior visual alignment falls below the safety threshold.
"""

import logging
from typing import TypedDict, Dict, Any, Tuple, List, Optional
from PIL import Image
import torch

from core.evaluator import FullStackBLIPEvaluator

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State schema maintained across LangGraph execution nodes."""
    image_path: str
    current_image: Image.Image
    salient_bbox: Tuple[float, float, float, float]
    best_caption: str
    best_s_vis_raw: float
    best_s_final: float
    candidates: List[Dict[str, Any]]
    dpo_pair: Optional[Dict[str, Any]]
    retry_count: int
    max_retries: int
    vis_threshold: float
    reflection_log: List[str]


class VeriVLMAgentWorkflow:
    """
    Closed-loop Multimodal Agent orchestrating FullStackBLIPEvaluator
    with localized region cropping and super-resolution re-perception.
    """

    def __init__(
        self,
        evaluator: FullStackBLIPEvaluator,
        transform: Any,
        device: torch.device,
        gamma: float = 0.35,
        num_beams: int = 5,
        diversity_penalty: float = 1.45,
        length_penalty: float = 1.0,
        dpo_export_path: Optional[str] = None,
    ) -> None:
        self.evaluator = evaluator
        self.transform = transform
        self.device = device
        self.gamma = gamma
        self.num_beams = num_beams
        self.diversity_penalty = diversity_penalty
        self.length_penalty = length_penalty
        self.dpo_export_path = dpo_export_path

    def perception_and_critic_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 1: Runs full-stack inference and evaluates posterior visual confidence."""
        tensor_img = self.transform(state["current_image"]).unsqueeze(0).to(self.device)
        result = self.evaluator.generate_full_stack(
            image=tensor_img,
            gamma=self.gamma,
            num_beams=self.num_beams,
            diversity_penalty=self.diversity_penalty,
            length_penalty=self.length_penalty,
            dpo_export_path=self.dpo_export_path,
        )

        step_idx = state["retry_count"]
        log_msg = (
            f"[Step {step_idx}] Generated: '{result['best_caption']}' "
            f"(S_vis_raw={result['best_s_vis_raw']:.4f}, S_final={result['best_s_final']:.4f})"
        )
        logger.info(log_msg)

        if step_idx > 0 and result["best_s_vis_raw"] < state["best_s_vis_raw"]:
            return {
                "reflection_log": state["reflection_log"]
                + [log_msg + " -> Retained initial prediction."]
            }

        return {
            "salient_bbox": result["salient_bbox"] if step_idx == 0 else state["salient_bbox"],
            "best_caption": result["best_caption"],
            "best_s_vis_raw": result["best_s_vis_raw"],
            "best_s_final": result["best_s_final"],
            "candidates": result["candidates"],
            "dpo_pair": result["dpo_pair"] or state.get("dpo_pair"),
            "reflection_log": state["reflection_log"] + [log_msg],
        }

    def adaptive_crop_zoom_tool_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Node 2 (Tool Calling): Crops the high-variance region identified by ViPER
        and applies bicubic upsampling for secondary fine-grained verification.
        """
        orig_img = Image.open(state["image_path"]).convert("RGB")
        width, height = orig_img.size
        x1_n, y1_n, x2_n, y2_n = state["salient_bbox"]

        crop_box = (
            int(x1_n * width),
            int(y1_n * height),
            int(x2_n * width),
            int(y2_n * height),
        )
        cropped_img = orig_img.crop(crop_box).resize((width, height), Image.Resampling.BICUBIC)

        tool_msg = (
            f"[Tool Call] Visual confidence ({state['best_s_vis_raw']:.4f}) below threshold "
            f"({state['vis_threshold']:.4f}). Cropped high-variance region {state['salient_bbox']}."
        )
        logger.info(tool_msg)

        return {
            "current_image": cropped_img,
            "retry_count": state["retry_count"] + 1,
            "reflection_log": state["reflection_log"] + [tool_msg],
        }

    @staticmethod
    def should_trigger_reflection(state: AgentState) -> str:
        """Conditional router guarding against low visual alignment and infinite loops."""
        if (
            state["best_s_vis_raw"] < state["vis_threshold"]
            and state["retry_count"] < state["max_retries"]
        ):
            return "invoke_zoom_tool"
        return "complete"

    def invoke(
        self, image_path: str, vis_threshold: float = 0.25, max_retries: int = 1
    ) -> AgentState:
        """
        Executes the state machine using LangGraph if installed, or falls back to
        a deterministic state loop with identical execution semantics.
        """
        initial_state: AgentState = {
            "image_path": image_path,
            "current_image": Image.open(image_path).convert("RGB"),
            "salient_bbox": (0.0, 0.0, 1.0, 1.0),
            "best_caption": "",
            "best_s_vis_raw": 0.0,
            "best_s_final": 0.0,
            "candidates": [],
            "dpo_pair": None,
            "retry_count": 0,
            "max_retries": max_retries,
            "vis_threshold": vis_threshold,
            "reflection_log": [],
        }

        try:
            from langgraph.graph import StateGraph, END

            graph = StateGraph(AgentState)
            graph.add_node("perception_and_critic", self.perception_and_critic_node)
            graph.add_node("zoom_tool", self.adaptive_crop_zoom_tool_node)
            graph.set_entry_point("perception_and_critic")
            graph.add_conditional_edges(
                "perception_and_critic",
                self.should_trigger_reflection,
                {"invoke_zoom_tool": "zoom_tool", "complete": END},
            )
            graph.add_edge("zoom_tool", "perception_and_critic")
            compiled_app = graph.compile()
            return compiled_app.invoke(initial_state)
        except ImportError:
            logger.info(
                "LangGraph package not detected; executing built-in deterministic state machine."
            )
            state = initial_state
            while True:
                state.update(self.perception_and_critic_node(state))
                if self.should_trigger_reflection(state) == "invoke_zoom_tool":
                    state.update(self.adaptive_crop_zoom_tool_node(state))
                else:
                    break
            return state