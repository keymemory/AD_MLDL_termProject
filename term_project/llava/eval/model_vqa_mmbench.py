import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

from PIL import Image
import math


all_options = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        visual_token_num=args.visual_token_num,
        important_ratio=args.important_ratio,
        enable_clustering=args.enable_clustering,
        stage1_tokens=args.stage1_tokens,
        merge_method=args.merge_method,
        kmeans_max_iter=args.kmeans_max_iter,
        taskaware_kd=args.taskaware_kd,
        selection_method=args.selection_method,
        energy_tau=args.energy_tau,
        stat_k=args.stat_k,
        stat_robust=args.stat_robust,
    )

    # Data
    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_bar = tqdm(questions.iterrows(), total=len(questions))
    for index, row in data_bar:
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image_tensor = process_images([image], image_processor, model.config)[0]

            with torch.inference_mode():
                output_ids, visual_token_num = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)
            data_bar.set_postfix({"visual_token_num": visual_token_num})

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
    ans_file.close()

    # [Phase2-B] adaptive selection 집계 → .meta/.dist (model_vqa_loader 방식 이식)
    _log = getattr(model, "_adaptive_log", [])
    if _log:
        _avg_n_imp = sum(x["n_imp"] for x in _log) / len(_log)
        _avg_M1 = sum(x["M1"] for x in _log) / len(_log)
        _avg_r = sum(x["n_imp"] / max(1, x["M1"]) for x in _log) / len(_log)
        _floor_pct = 100.0 * sum(1 for x in _log if x.get("floor")) / len(_log)
        _cap_pct = 100.0 * sum(1 for x in _log if x.get("cap")) / len(_log)
        with open(answers_file + ".meta", "w") as _mf:
            _mf.write(json.dumps({
                "selection_method": args.selection_method,
                "count": len(_log),
                "avg_n_imp": round(_avg_n_imp, 3),
                "avg_M1": round(_avg_M1, 3),
                "avg_r": round(_avg_r, 4),
                "floor_pct": round(_floor_pct, 1),
                "cap_pct": round(_cap_pct, 1),
            }) + "\n")
        with open(answers_file + ".dist.jsonl", "w") as _df:
            for _rec in _log:
                _df.write(json.dumps(_rec) + "\n")
        print(f"[Phase2] mmbench adaptive: avg_M1={_avg_M1:.2f} floor={_floor_pct:.0f}% n={len(_log)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--visual_token_num", type=int, default=576)
    parser.add_argument("--important_ratio", type=float, default=0.5)
    # [Two-Stage] Stage2 clustering + [Phase1] adaptive selection
    parser.add_argument("--enable_clustering", action="store_true", default=False)
    parser.add_argument("--stage1_tokens", type=int, default=None)
    parser.add_argument("--merge_method", type=str, default="simple_avg",
                        choices=["simple_avg", "weighted_avg", "taskaware"])
    parser.add_argument("--taskaware_kd", type=float, default=1.5)
    parser.add_argument("--kmeans_max_iter", type=int, default=10)
    parser.add_argument("--selection_method", type=str, default="topk",
                        choices=["topk", "energy", "statistical"])
    parser.add_argument("--energy_tau", type=float, default=0.5)
    parser.add_argument("--stat_k", type=float, default=2.0)
    parser.add_argument("--stat_robust", action="store_true", default=False)
    args = parser.parse_args()

    eval_model(args)
