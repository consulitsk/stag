import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    quantize_dynamic(
        model_input=args.inp,
        model_output=args.out,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
