import argparse
import numpy as np
import onnxruntime as ort

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    # ORT
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])

    # Input
    img = np.random.randn(1, 3, args.image_size, args.image_size).astype(np.float32)

    # Run ORT
    ort_out = sess.run(None, {"image": img})[0]

    print("ORT output shape:", ort_out.shape, "dtype:", ort_out.dtype)
    print("ORT stats:", float(np.min(ort_out)), float(np.max(ort_out)), float(np.mean(ort_out)))

if __name__ == "__main__":
    main()
