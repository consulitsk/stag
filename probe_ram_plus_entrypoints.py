import argparse, inspect, time, re
import numpy as np
import torch
import psutil
from huggingface_hub import hf_hub_download

def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)

def build_model(repo_id: str, filename: str, image_size: int, vit: str):
    ckpt = hf_hub_download(repo_id=repo_id, filename=filename)
    from ram.models import ram_plus
    model = ram_plus(pretrained=ckpt, image_size=image_size, vit=vit)
    model.eval()
    return model, ckpt

def is_tensorish(x):
    return isinstance(x, torch.Tensor)

def extract_tensor(out):
    # Prefer a 2D logits-like tensor if present
    if is_tensorish(out):
        return out
    if isinstance(out, (list, tuple)):
        for t in out:
            if is_tensorish(t):
                return t
    if isinstance(out, dict):
        for k in ("logits", "pred_logits", "tag_logits", "scores", "probs"):
            if k in out and is_tensorish(out[k]):
                return out[k]
        for v in out.values():
            if is_tensorish(v):
                return v
    return None

def try_call(fn, image):
    try:
        with torch.inference_mode():
            out = fn(image)
        t = extract_tensor(out)
        return t, None
    except Exception as e:
        return None, e

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="xinyu1205/recognize-anything-plus-model")
    ap.add_argument("--filename", default="ram_plus_swin_large_14m.pth")
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--vit", default="swin_l")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--interop", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop)

    print("RSS start (MB):", rss_mb())
    model, ckpt = build_model(args.repo_id, args.filename, args.image_size, args.vit)
    print("[ckpt]", ckpt)
    print("RSS after model load (MB):", rss_mb())

    image = torch.randn(1, 3, args.image_size, args.image_size)

    # 1) Skús prečítať, čo používa inference funkcia (ak existuje)
    hinted = []
    try:
        from ram import inference_ram
        src = inspect.getsource(inference_ram)
        hinted = re.findall(r"model\.([A-Za-z_][A-Za-z0-9_]*)\(", src)
        hinted = list(dict.fromkeys(hinted))  # unique, keep order
        if hinted:
            print("[hint] inference_ram references:", hinted)
    except Exception as e:
        print("[hint] cannot inspect ram.inference_ram:", e)

    # 2) Kandidáti: najprv hinty + typické názvy
    preferred = [
        "inference", "predict", "forward_inference", "forward_test",
        "generate_tag", "tagging", "forward_tag", "forward_tags",
        "get_logits", "get_tag_logits", "encode_image", "visual_forward",
    ]
    candidates = []
    for name in hinted + preferred + dir(model):
        if name.startswith("_"):
            continue
        if name in candidates:
            continue
        attr = getattr(model, name, None)
        if callable(attr):
            candidates.append(name)

    print("forward signature:", inspect.signature(model.forward))

    ok = []
    for name in candidates:
        fn = getattr(model, name)
        # rýchly filter: preferuj metódy, ktoré akceptujú 1 argument (image)
        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            # bound method => 'image' je prvý parameter
            # allow *args/**kwargs too; we will try_call anyway
        except Exception:
            sig = None

        t, err = try_call(fn, image)
        if t is not None and isinstance(t, torch.Tensor):
            ok.append((name, tuple(t.shape), str(t.dtype)))
            print(f"[OK] {name} -> shape={tuple(t.shape)} dtype={t.dtype}")
            # netestuj všetko do nekonečna
            if len(ok) >= 15:
                break

    if not ok:
        print("\n[FAIL] Nenašiel som žiadnu metódu, ktorá by zobrala iba image a vrátila tensor.")
        print("Ďalší krok: exportovať konkrétnu internú inferenčnú vetvu (logits) podľa zdrojáku ram.inference_ram.")
    else:
        print("\nTop candidates (name, shape, dtype):")
        for row in ok[:10]:
            print(" ", row)

if __name__ == "__main__":
    main()
