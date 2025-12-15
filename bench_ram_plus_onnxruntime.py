import argparse, time
import numpy as np
import psutil
import onnxruntime as ort

def rss_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)

def stats_ms(samples):
    a = np.array(samples, dtype=np.float64)
    return {
        "avg_ms": float(np.mean(a)),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(np.min(a)),
        "max_ms": float(np.max(a)),
    }

def make_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

def bench(sess, image, warmup, iters):
    # warmup
    for _ in range(warmup):
        _ = sess.run(None, {"image": image})
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = sess.run(None, {"image": image})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return stats_ms(times)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-fp32", required=True)
    # The INT8 benchmark is disabled because the current ONNX Runtime CPU provider
    # does not support the 'ConvInteger' operator produced by the quantization process.
    # ap.add_argument("--onnx-int8", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    print("=== ORT ENV ===")
    print("onnxruntime:", ort.__version__)
    print("providers:", ort.get_available_providers())
    print("threads:", args.threads)
    print("RSS start (MB):", rss_mb())

    img = np.random.randn(1, 3, args.image_size, args.image_size).astype(np.float32)

    sess_fp32 = make_session(args.onnx_fp32, args.threads)
    print("RSS after FP32 session (MB):", rss_mb())
    r_fp32 = bench(sess_fp32, img, args.warmup, args.iters)
    print("ORT FP32:", r_fp32)
    print("RSS after FP32 bench (MB):", rss_mb())

if __name__ == "__main__":
    main()
