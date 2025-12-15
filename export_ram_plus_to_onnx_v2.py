import argparse, inspect, re, os
import torch
import psutil
from huggingface_hub import hf_hub_download

def rss_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)

def build_model(repo_id: str, filename: str, image_size: int, vit: str):
    ckpt = hf_hub_download(repo_id=repo_id, filename=filename)
    from ram.models import ram_plus
    model = ram_plus(pretrained=ckpt, image_size=image_size, vit=vit)
    model.eval()
    return model, ckpt

def extract_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)):
        for t in out:
            if isinstance(t, torch.Tensor):
                return t
    if isinstance(out, dict):
        for k in ("logits", "pred_logits", "tag_logits", "scores", "probs"):
            if k in out and isinstance(out[k], torch.Tensor):
                return out[k]
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v
    return None

class Wrapper(torch.nn.Module):
    def __init__(self, model, method_name: str):
        super().__init__()
        self.model = model
        self.method_name = method_name

    def forward(self, image):
        fn = getattr(self.model, self.method_name)
        out = fn(image)
        t = extract_tensor(out)
        if t is None:
            raise RuntimeError(f"Entry {self.method_name} did not return a Tensor-like output.")
        return t

def pick_entrypoint(model):
    hinted = []
    try:
        from ram import inference_ram
        src = inspect.getsource(inference_ram)
        hinted = re.findall(r"model\.([A-Za-z_][A-Za-z0-9_]*)\(", src)
        hinted = list(dict.fromkeys(hinted))
    except Exception:
        pass

    preferred = [
        "visual_encoder",
        "generate_tag", "forward_tag", "forward_tags", "tagging",
        "predict", "inference", "forward_inference", "forward_test",
        "get_logits", "get_tag_logits",
    ]
    tried = []
    for name in hinted + preferred + dir(model):
        if name.startswith("_"):
            continue
        if name in tried:
            continue
        tried.append(name)
        attr = getattr(model, name, None)
        if not callable(attr):
            continue
        # Quick probe
        try:
            with torch.inference_mode():
                dummy = torch.randn(1, 3, 384, 384)
                out = attr(dummy)
                t = extract_tensor(out)
                if isinstance(t, torch.Tensor):
                    return name
        except Exception:
            continue
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="xinyu1205/recognize-anything-plus-model")
    ap.add_argument("--filename", default="ram_plus_swin_large_14m.pth")
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--vit", default="swin_l")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--onnx-out", default="ram_plus_fp32.onnx")
    ap.add_argument("--entrypoint", default="auto")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--interop", type=int, default=1)
    ap.add_argument("--external-data", action="store_true", help="store large weights in external .data file")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop)

    print("RSS start (MB):", rss_mb())
    model, ckpt = build_model(args.repo_id, args.filename, args.image_size, args.vit)
    print("[ckpt]", ckpt)
    print("forward signature:", inspect.signature(model.forward))
    print("RSS after model load (MB):", rss_mb())

    if args.entrypoint == "auto":
        ep = pick_entrypoint(model)
        if ep is None:
            raise RuntimeError(
                "Nenašiel som image-only entrypoint. Spusť probe_ram_plus_entrypoints.py "
                "a pošli mi [OK] kandidátov, alebo nastav --entrypoint na konkrétnu metódu."
            )
        print("[entrypoint] auto-picked:", ep)
    else:
        ep = args.entrypoint
        print("[entrypoint] user:", ep)

    export_target = getattr(model, ep)
    if isinstance(export_target, torch.nn.Module):
        print(f"[info] Entrypoint '{ep}' is an nn.Module, exporting it directly.")
        export_model = export_target.eval()
    else:
        print(f"[info] Entrypoint '{ep}' is not an nn.Module, using wrapper.")
        export_model = Wrapper(model, ep).eval()

    image = torch.randn(1, 3, args.image_size, args.image_size)
    out_path = args.onnx_out

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    output_names = ["features"]
    dynamic_axes = {"image": {0: "batch"}, output_names[0]: {0: "batch"}}

    print("[export] torch.onnx.export ...")
    torch.onnx.export(
        export_model,
        (image,),
        out_path,
        opset_version=args.opset,
        input_names=["image"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        export_params=True,
        use_external_data_format=bool(args.external_data),
    )
    print("[export] wrote:", out_path)
    print("RSS end (MB):", rss_mb())

if __name__ == "__main__":
    main()
