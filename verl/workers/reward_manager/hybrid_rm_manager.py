# Copyright 2025
# Custom hybrid reward manager for:
#   final_reward = rule_reward + rm_weight * reward_model_score
#
# Assumptions:
# 1. rule reward still comes from custom_reward_function via compute_score
# 2. reward model is a HuggingFace AutoModelForSequenceClassification
# 3. scalar RM score is added to the last valid token of each sample
#
# If your DataProto batch field names differ, adjust _extract_prompt_response_texts().

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from verl.workers.reward_manager import get_reward_manager_cls, register


@register("hybrid_rm")
class HybridRMRewardManager:
    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score,
        reward_fn_key,
        rm_model_path: str,
        rm_tokenizer_path: str | None = None,
        rm_batch_size: int = 4,
        rm_max_length: int = 2048,
        rm_weight: float = 1.0,
        rm_input_template: str = "{prompt}\n\n{response}",
        rm_output_mode: str = "auto",
        rm_device: str | None = None,
        base_reward_manager: str = "naive",
        **kwargs,
    ):
        """
        Args:
            tokenizer: verl trainer tokenizer
            num_examine: inherited from verl
            compute_score: rule-based reward function loaded by get_custom_reward_fn()
            reward_fn_key: inherited from verl
            rm_model_path: path to trained scoring model
            rm_tokenizer_path: optional tokenizer path for RM
            rm_batch_size: batch size for RM forward
            rm_max_length: max length for RM input
            rm_weight: final score = rule_reward + rm_weight * rm_score
            rm_input_template: text format fed into RM
            rm_output_mode:
                - "auto": infer from logits shape
                - "scalar": logits.squeeze(-1)
                - "pos_prob": softmax(logits)[:, 1]
                - "pos_logit": logits[:, 1]
                - "margin": logits[:, 1] - logits[:, 0]
            rm_device: e.g. "cuda:0", "cpu", None(auto)
            base_reward_manager: use naive/prime/batch/dapo as underlying rule reward manager
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

        self.rm_model_path = rm_model_path
        self.rm_tokenizer_path = rm_tokenizer_path or rm_model_path
        self.rm_batch_size = rm_batch_size
        self.rm_max_length = rm_max_length
        self.rm_weight = rm_weight
        self.rm_input_template = rm_input_template
        self.rm_output_mode = rm_output_mode

        self.rm_device = rm_device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 底层 rule reward manager，仍然走你现在已有的 custom_reward_function
        base_cls = get_reward_manager_cls(base_reward_manager)
        self.base_manager = base_cls(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            reward_fn_key=reward_fn_key,
            **kwargs,
        )

        # 加载评分模型
        self.rm_tokenizer = AutoTokenizer.from_pretrained(self.rm_tokenizer_path, trust_remote_code=True)
        if self.rm_tokenizer.pad_token is None:
            self.rm_tokenizer.pad_token = self.rm_tokenizer.eos_token

        self.rm_model = AutoModelForSequenceClassification.from_pretrained(
            self.rm_model_path,
            trust_remote_code=True,
        )
        self.rm_model.eval()
        self.rm_model.to(self.rm_device)

    def __call__(self, data, return_dict: bool = False):
        """
        Returns:
            if return_dict:
                {
                  "reward_tensor": ...,
                  "reward_extra_info": ...
                }
            else:
                reward_tensor
        """
        # 先拿原有规则奖励
        try:
            base_result = self.base_manager(data, return_dict=True)
            reward_tensor = base_result["reward_tensor"]
            reward_extra_info = base_result.get("reward_extra_info", {})
        except TypeError:
            reward_tensor = self.base_manager(data)
            reward_extra_info = {}

        # 再算 RM 分数
        prompts, responses = self._extract_prompt_response_texts(data)
        rm_scores = self._score_with_reward_model(prompts, responses)

        # 把 RM 分数加到 reward tensor 上
        reward_tensor = self._add_scalar_rewards_to_reward_tensor(
            reward_tensor=reward_tensor,
            rm_scores=rm_scores,
            data=data,
        )

        reward_extra_info = dict(reward_extra_info)
        reward_extra_info["rm_scores"] = rm_scores.detach().cpu().tolist()
        reward_extra_info["rm_weight"] = self.rm_weight

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor

    @torch.no_grad()
    def _score_with_reward_model(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        texts = [
            self.rm_input_template.format(prompt=p, response=r)
            for p, r in zip(prompts, responses)
        ]

        all_scores = []
        for start in range(0, len(texts), self.rm_batch_size):
            batch_texts = texts[start : start + self.rm_batch_size]
            inputs = self.rm_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.rm_max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.rm_device) for k, v in inputs.items()}

            outputs = self.rm_model(**inputs)
            logits = outputs.logits

            scores = self._convert_logits_to_scores(logits)
            all_scores.append(scores.detach().float().cpu())

        rm_scores = torch.cat(all_scores, dim=0)
        return rm_scores * self.rm_weight

    def _convert_logits_to_scores(self, logits: torch.Tensor) -> torch.Tensor:
        # 常见情况 1: regression / single score
        if self.rm_output_mode == "scalar":
            return logits.squeeze(-1)

        # 常见情况 2: binary classifier
        if self.rm_output_mode == "pos_prob":
            return torch.softmax(logits, dim=-1)[:, 1]

        if self.rm_output_mode == "pos_logit":
            return logits[:, 1]

        if self.rm_output_mode == "margin":
            return logits[:, 1] - logits[:, 0]

        # auto 模式
        if logits.ndim == 2 and logits.shape[-1] == 1:
            return logits.squeeze(-1)

        if logits.ndim == 2 and logits.shape[-1] >= 2:
            # 默认用正类 margin，更稳一些
            return logits[:, 1] - logits[:, 0]

        return logits.view(-1)

    def _extract_prompt_response_texts(self, data) -> tuple[list[str], list[str]]:
        """
        尽量兼容不同 batch 字段命名。
        优先级：
        1. 已经是字符串字段
        2. token ids 解码
        """
        batch = self._get_batch_dict(data)

        # 1) 直接拿字符串
        prompt_candidates = ["prompts", "prompt", "prompt_text", "query"]
        response_candidates = ["responses", "response", "response_text", "answer"]

        prompt_texts = self._try_get_text_list(batch, prompt_candidates)
        response_texts = self._try_get_text_list(batch, response_candidates)

        if prompt_texts is not None and response_texts is not None:
            return prompt_texts, response_texts

        # 2) 尝试从 token ids 解码
        prompt_id_candidates = ["prompt_token_ids", "prompts", "input_ids"]
        response_id_candidates = ["response_token_ids", "responses", "output_ids"]

        prompt_ids = self._try_get_tensor_or_list(batch, prompt_id_candidates)
        response_ids = self._try_get_tensor_or_list(batch, response_id_candidates)

        if prompt_ids is None or response_ids is None:
            available_keys = list(batch.keys())
            raise KeyError(
                "Failed to extract prompt/response from DataProto.batch. "
                f"Available keys: {available_keys}"
            )

        prompt_texts = self._decode_batch_ids(prompt_ids)
        response_texts = self._decode_batch_ids(response_ids)
        return prompt_texts, response_texts

    def _add_scalar_rewards_to_reward_tensor(self, reward_tensor, rm_scores: torch.Tensor, data):
        """
        把每个 sample 的 rm scalar score 加到最后一个有效 token 上。
        兼容：
        - reward_tensor: [B]
        - reward_tensor: [B, T]
        """
        if not torch.is_tensor(reward_tensor):
            reward_tensor = torch.tensor(reward_tensor)

        device = reward_tensor.device
        dtype = reward_tensor.dtype
        rm_scores = rm_scores.to(device=device, dtype=dtype)

        if reward_tensor.ndim == 1:
            return reward_tensor + rm_scores

        if reward_tensor.ndim != 2:
            raise ValueError(f"Unsupported reward_tensor shape: {tuple(reward_tensor.shape)}")

        B, T = reward_tensor.shape
        if rm_scores.shape[0] != B:
            raise ValueError(
                f"Batch size mismatch: reward_tensor batch={B}, rm_scores batch={rm_scores.shape[0]}"
            )

        last_indices = self._infer_last_response_token_indices(data, T, device=device)

        out = reward_tensor.clone()
        out[torch.arange(B, device=device), last_indices] += rm_scores
        return out

    def _infer_last_response_token_indices(self, data, T: int, device):
        batch = self._get_batch_dict(data)

        # 优先尝试 response_attention_mask
        for key in ["response_attention_mask", "responses_attention_mask", "attention_mask"]:
            if key in batch:
                mask = batch[key]
                if not torch.is_tensor(mask):
                    mask = torch.tensor(mask)
                mask = mask.to(device)

                if mask.ndim == 2:
                    if mask.shape[1] > T:
                        mask = mask[:, -T:]
                    valid_lens = mask.long().sum(dim=1).clamp(min=1)
                    return valid_lens - 1

        # 再尝试 response token ids != pad
        for key in ["response_token_ids", "responses"]:
            if key in batch:
                ids = batch[key]
                if not torch.is_tensor(ids):
                    ids = torch.tensor(ids)
                ids = ids.to(device)

                if ids.ndim == 2:
                    if ids.shape[1] > T:
                        ids = ids[:, -T:]
                    pad_id = self.tokenizer.pad_token_id
                    if pad_id is None:
                        pad_id = 0
                    valid_lens = (ids != pad_id).long().sum(dim=1).clamp(min=1)
                    return valid_lens - 1

        # 实在推不出来，就默认加到最后一个位置
        # 从 batch 中推 batch size
        for key in ["responses", "response_token_ids", "response_attention_mask", "attention_mask"]:
            if key in batch:
                x = batch[key]
                if torch.is_tensor(x):
                    bsz = x.shape[0]
                else:
                    bsz = len(x)
                return torch.full((bsz,), T - 1, device=device, dtype=torch.long)

        raise ValueError("Cannot infer batch size for reward placement.")

    def _decode_batch_ids(self, batch_ids) -> list[str]:
        texts = []
        for ids in batch_ids:
            if torch.is_tensor(ids):
                ids = ids.detach().cpu().tolist()
            if isinstance(ids, tuple):
                ids = list(ids)
            texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))
        return texts

    def _try_get_text_list(self, batch: dict[str, Any], keys: list[str]) -> list[str] | None:
        for key in keys:
            if key not in batch:
                continue
            val = batch[key]
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], str):
                return val
        return None

    def _try_get_tensor_or_list(self, batch: dict[str, Any], keys: list[str]):
        for key in keys:
            if key in batch:
                return batch[key]
        return None

    def _get_batch_dict(self, data) -> dict[str, Any]:
        if hasattr(data, "batch"):
            return data.batch
        if isinstance(data, dict) and "batch" in data:
            return data["batch"]
        raise TypeError("Unsupported DataProto format: cannot find batch field")