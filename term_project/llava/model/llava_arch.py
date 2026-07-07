#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
import json
import os

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


_EPS = 1e-8


def _l2norm(x, dim=-1):
    return x / x.float().norm(dim=dim, keepdim=True).clamp_min(_EPS).to(x.dtype)


def _max_distance_knee(y):
    """Kneedle fallback: index with max distance from first-last chord."""
    if y.numel() <= 2:
        return int(y.numel())
    yf = y.float()
    x = torch.linspace(0, 1, yf.numel(), device=y.device)
    y0, y1 = yf[0], yf[-1]
    baseline = y0 + (y1 - y0) * x
    idx = int((yf - baseline).abs().argmax().item())
    return max(1, idx + 1)


def _attention_gain_elbow_indices(attn):
    order = attn.argsort(descending=True)
    sorted_attn = attn[order].float()
    if sorted_attn.numel() <= 2:
        return order[:1], 1, torch.empty(0, device=attn.device)
    gains = sorted_attn[:-1] - sorted_attn[1:]
    k = _max_distance_knee(gains)
    k = max(1, min(k, order.numel()))
    return order[:k], k, gains


def _fixed_diverse_indices(features_norm, residual_indices, diverse_k):
    B = features_norm.shape[0]
    while diverse_k > 0:
        R = residual_indices.shape[1]
        r = min(8, R - diverse_k)
        if r <= 0:
            break

        residual_tokens = features_norm[
            torch.arange(B, device=features_norm.device).unsqueeze(-1).expand(-1, R),
            residual_indices
        ]
        a, b = residual_tokens[..., ::2, :], residual_tokens[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        scores = scores.max(dim=-1).values
        distinct_indices = scores.argsort(dim=-1, descending=True)[:, r:]
        keep = distinct_indices.shape[1]
        residual_indices = torch.cat([
            residual_indices[..., ::2][
                torch.arange(B, device=features_norm.device).unsqueeze(-1).expand(-1, keep),
                distinct_indices
            ],
            residual_indices[..., 1::2]
        ], dim=-1)
    return residual_indices[:, :diverse_k]


def _greedy_cosine_gain_elbow_indices(features_norm, important_idx):
    device = features_norm.device
    n = features_norm.shape[0]
    important_mask = torch.zeros(n, dtype=torch.bool, device=device)
    important_mask[important_idx] = True
    candidates = torch.nonzero(~important_mask, as_tuple=False).flatten()
    if candidates.numel() == 0:
        return candidates, 0, torch.empty(0, device=device)

    selected = important_idx.clone()
    order = []
    gains = []
    cand = candidates
    while cand.numel() > 0:
        sim = features_norm[cand].float() @ features_norm[selected].float().t()
        max_sim = sim.max(dim=1).values
        pos = int(max_sim.argmin().item())
        picked = cand[pos]
        order.append(picked)
        gains.append(1.0 - max_sim[pos])
        selected = torch.cat([selected, picked.view(1)])
        cand = cand[torch.arange(cand.numel(), device=device) != pos]

    if not order:
        return candidates[:0], 0, torch.empty(0, device=device)
    order = torch.stack(order)
    gains = torch.stack(gains)
    k = _max_distance_knee(gains)
    k = max(0, min(k, order.numel()))
    return order[:k], k, gains


class LlavaMetaModel:

    def __init__(self, config, **kwargs):
        super(LlavaMetaModel, self).__init__(config, **kwargs)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    # [VisPruner] Generate index masks using visual cues
    def encode_images(self, images):
        image_features, image_attentions = self.get_model().get_vision_tower()(images, output_attentions=True) # (B, N, C), (B, H, N)
        
        B, N, C = image_features.shape
        device = image_features.device
        index_masks = torch.ones(B, N, dtype=torch.bool, device=device)

        # [Two-Stage] Stage1 보존 토큰 수 M1, 최종 토큰 수 M2
        M2 = self.get_visual_token_num()
        enable_clustering = self.get_enable_clustering()
        M1 = self.get_stage1_tokens() if enable_clustering else M2
        # Stage1 선택은 M1개 기준으로 수행 (clustering off면 M1==M2 → 기존 VisPruner와 동일)
        visual_token_num = M1
        important_ratio = self.get_important_ratio() # r
        important_token_num = int(visual_token_num * important_ratio) # T_imp = M1 * r
        diverse_token_num = visual_token_num - important_token_num # T_div = M1 * (1 - r)
        select_mode = self.get_select_mode()
        diverse_mode = self.get_diverse_mode()

        # [VisPruner] Select important tokens using attention scores
        image_attentions = image_attentions.mean(dim=1) # (B, N)
        image_normalized = image_features / image_features.norm(dim=-1, keepdim=True) # (B, N, C)

        # [Exp2] Stage 1 선택 자동화. fixed/fixed일 때는 기존 VisPruner 경로와 동일.
        selected_per_batch = []
        stats_per_batch = []
        if select_mode == "fixed" and diverse_mode == "fixed":
            token_indices = image_attentions.argsort(dim=-1, descending=True) # (B, N)
            important_indices = token_indices[:, :important_token_num] # (B, T_imp)
            residual_indices = token_indices[:, important_token_num:] # (B, N - T_imp)
            diverse_indices = _fixed_diverse_indices(image_normalized, residual_indices, diverse_token_num)
            selected_indices = torch.cat([important_indices, diverse_indices], dim=-1) if diverse_token_num > 0 else important_indices
            selected_per_batch = [selected_indices[b] for b in range(B)]
            for b in range(B):
                stats_per_batch.append({
                    "select_mode": select_mode, "diverse_mode": diverse_mode,
                    "important": int(important_indices.shape[1]),
                    "diverse": int(selected_indices.shape[1] - important_indices.shape[1]),
                    "m1": int(selected_indices.shape[1]), "m2": int(M2),
                })
        elif select_mode == "attngain" and diverse_mode == "greedygain":
            for b in range(B):
                imp, n_imp, imp_gains = _attention_gain_elbow_indices(image_attentions[b])
                div, n_div, div_gains = _greedy_cosine_gain_elbow_indices(image_normalized[b], imp)
                selected = torch.cat([imp, div], dim=0).unique(sorted=False)
                if selected.numel() < M2:
                    order = image_attentions[b].argsort(descending=True)
                    need = M2 - selected.numel()
                    mask = torch.ones(N, dtype=torch.bool, device=device)
                    mask[selected] = False
                    selected = torch.cat([selected, order[mask[order]][:need]])
                selected = selected[:N]
                selected_per_batch.append(selected)
                stats_per_batch.append({
                    "select_mode": select_mode, "diverse_mode": diverse_mode,
                    "important": int(imp.numel()), "diverse": int(div.numel()),
                    "m1": int(selected.numel()), "m2": int(M2),
                    "attn_gain_elbow_k": int(n_imp),
                    "diverse_gain_elbow_k": int(n_div),
                    "attn_gain_first": float(imp_gains[0].item()) if imp_gains.numel() else 0.0,
                    "diverse_gain_first": float(div_gains[0].item()) if div_gains.numel() else 0.0,
                })
        else:
            raise ValueError(
                f"unsupported selection pair: {select_mode}/{diverse_mode}; "
                "use fixed/fixed or attngain/greedygain"
            )

        index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b, selected in enumerate(selected_per_batch):
            index_masks[b, selected] = True

        log_path = os.environ.get("EXP2_SELECTION_LOG")
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a") as f:
                for row in stats_per_batch:
                    f.write(json.dumps(row) + "\n")

        # [VisPruner] mm_projector 적용 (B, N, C) -> (B, N, D_llm)
        image_features = self.get_model().mm_projector(image_features)

        # [Two-Stage] Stage 2: Spherical K-Means 병합 (clustering on & M2 < M1)
        if enable_clustering:
            from llava.model.spherical_kmeans import merge_tokens
            merge_method = self.get_merge_method()
            kmeans_max_iter = self.get_kmeans_max_iter()
            merged_list = []
            for b in range(B):
                # Stage1 보존 토큰 (M1, D) 및 토큰별 attention score (M1,)
                cur_idx = selected_per_batch[b]
                feats_b = image_features[b][cur_idx]                  # (M1, D)
                attn_b = image_attentions[b][cur_idx]                 # (M1,)
                if M2 < feats_b.shape[0]:
                    merged_b = merge_tokens(
                        feats_b, attn_b, M2,
                        method=merge_method, max_iter=kmeans_max_iter,
                    )                                                 # (M2, D)
                else:
                    merged_b = feats_b
                merged_list.append(merged_b)
            merged = torch.stack(merged_list, dim=0)                  # (B, M2, D)
            # index_masks=None → 이미 최종 토큰. prepare_inputs에서 그대로 사용
            return merged, None

        return image_features, index_masks

    # [VisPruner] Prune visual tokens according to index masks
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, modalities=["image"], image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # [VisPruner] Prune visual tokens
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features, index_masks = self.encode_images(concat_images)
            if index_masks is None:
                raise NotImplementedError(
                    "[Two-Stage] clustering은 단일 이미지 경로(LLaVA-1.5)에서만 지원됩니다. "
                    "multi-image/anyres(LLaVA-NeXT)에서는 --enable_clustering 미사용."
                )
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            index_masks = torch.split(index_masks, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            mm_patch_merge_type = mm_patch_merge_type.replace('_unpad', '')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
                index_masks = [x.flatten(0, 1) for x in index_masks]
                image_features = [x[m] for x, m in zip(image_features, index_masks)]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, (image_feature, index_mask) in enumerate(zip(image_features, index_masks)):
                    if image_feature.shape[0] > 1:
                        base_image_feature, base_index_mask = image_feature[0], index_mask[0]
                        image_feature, index_mask = image_feature[1:], index_mask[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            index_mask = index_mask.view(num_patch_height, num_patch_width, height, width)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            index_mask = index_mask.permute(0, 2, 1, 3).contiguous().unsqueeze(0)
                            index_mask = index_mask.flatten(1, 2).flatten(2, 3)
                            index_mask = unpad_image(index_mask, image_sizes[image_idx])
                            index_mask = torch.cat((
                                index_mask,
                                torch.ones(*index_mask.shape[:-1], 1, dtype=torch.bool).to(index_mask.device)
                            ), dim=-1)
                            index_mask = index_mask.flatten(1, 2).squeeze(0)
                            image_feature = image_feature[index_mask]
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                            index_mask = index_mask.permute(0, 2, 1, 3).contiguous()
                            index_mask = index_mask.flatten(0, 3)
                            image_feature = image_feature[index_mask]
                        base_image_feature = base_image_feature[base_index_mask]
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        index_mask = index_mask[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                            index_mask = torch.cat((
                                index_mask,
                                torch.ones(1, dtype=torch.bool).to(index_mask.device)
                            ), dim=0)
                        image_feature = image_feature[index_mask]
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features, index_masks = self.encode_images(images)
            if index_masks is None:
                # [Two-Stage] 이미 Stage2 병합 완료 (B, M2, D) → 시퀀스로 평탄화
                image_features = image_features.flatten(0, 1).unsqueeze(0)
            else:
                image_features = image_features[index_masks].unsqueeze(0)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, image_features[0].shape[0]

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
