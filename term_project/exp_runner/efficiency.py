"""실험5: 효율성 측정 — POPE 110샘플(앞10 warmup, 100 평균)."""
import os, json, time, torch
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from PIL import Image

MP = "models/llava-v1.5-7b"
P = "playground/data/eval/pope"
rows = [json.loads(l) for l in open(f"{P}/llava_pope_test.jsonl")][:110]

# (id, M2, clustering, M1, method)
SETTINGS = [
    ("A-64",  64, False, 64,  "simple_avg"),
    ("B-64s", 64, True, 128, "simple_avg"),
    ("B-64w", 64, True, 128, "weighted_avg"),
    ("A-32",  32, False, 32,  "simple_avg"),
    ("B-32s", 32, True,  64,  "simple_avg"),
]
print(f"{'Setting':<8}{'M2':>4}{'Clust':>6}{'RedRatio%':>10}{'Latency(s/q)':>14}{'GPUmem(GB)':>12}")
for sid, m2, clust, m1, method in SETTINGS:
    name = get_model_name_from_path(MP)
    tok, model, ip, _ = load_pretrained_model(
        MP, None, name, visual_token_num=m2, important_ratio=0.5,
        enable_clustering=clust, stage1_tokens=m1, merge_method=method, kmeans_max_iter=10)
    torch.cuda.reset_peak_memory_stats()
    lat = []
    for i, r in enumerate(rows):
        img = Image.open(os.path.join(f"{P}/val2014", r["image"])).convert("RGB")
        it = process_images([img], ip, model.config)[0].unsqueeze(0).half().cuda()
        qs = DEFAULT_IMAGE_TOKEN + "\n" + r["text"]
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], qs); conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        torch.cuda.synchronize(); t0 = time.time()
        with torch.inference_mode():
            model.generate(ids, images=it, image_sizes=[img.size], do_sample=False,
                           temperature=0, max_new_tokens=16, use_cache=True)
        torch.cuda.synchronize()
        if i >= 10:  # warmup 제외
            lat.append(time.time() - t0)
    mem = torch.cuda.max_memory_allocated() / 1e9
    avg = sum(lat) / len(lat)
    red = (576 - m2) / 576 * 100
    print(f"{sid:<8}{m2:>4}{str(clust):>6}{red:>10.1f}{avg:>14.4f}{mem:>12.2f}", flush=True)
    del model; torch.cuda.empty_cache()
print("EFF_DONE")
