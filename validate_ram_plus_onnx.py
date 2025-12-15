import argparse
import numpy as np
import torch
from huggingface_hub import hf_hub_download
import onnxruntime as ort

def build_model(repo_id, filename, image_size, vit):
    ckpt = hf_hub_download(repo_id=repo_id, filename=filename)
    from ram.models import ram_plus
    model = ram_plus(pretrained=ckpt, image_size=image_size, vit=vit)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--repo-id", default="xinyu1205/recognize-anything-plus-model")
    ap.add_argument("--filename", default="ram_plus_swin_large_14m.pth")
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--vit", default="swin_l")
    ap.add_argument("--entrypoint", default="visual_encoder")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    # PyTorch
    model = build_model(args.repo_id, args.filename, args.image_size, args.vit)
    entrypoint = getattr(model, args.entrypoint)

    # ORT
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])

    # Input
    img_np = np.random.randn(1, 3, args.image_size, args.image_size).astype(np.float32)
    img_torch = torch.from_numpy(img_np)

    # Run ORT
    ort_out = sess.run(None, {"image": img_np})[0]

    # Run PyTorch
    with torch.no_grad():
        torch_out = entrypoint(img_torch).numpy()

    # Compare
    print("PyTorch output shape:", torch_out.shape, "dtype:", torch_out.dtype)
    print("ORT output shape:", ort_out.shape, "dtype:", ort_out.dtype)

    if np.allclose(torch_out, ort_out, atol=1e-2):
        print("\n✅ Outputs are numerically close enough.")
    else:
        print("\n❌ Outputs are NOT numerically close enough.")

    print("\nPyTorch stats:", float(np.min(torch_out)), float(np.max(torch_out)), float(np.mean(torch_out)))
    print("ORT stats:", float(np.min(ort_out)), float(np.max(ort_out)), float(np.mean(ort_out)))


if __name__ == "__main__":
    main()
