#!/usr/bin/env python3
import argparse
import time
import numpy as np

def normalize_shape(shape):
    # 把 -1/None 替为 1
    return tuple((1 if (s is None or (isinstance(s, int) and s < 0)) else int(s)) for s in shape)

def make_random_input(shape):
    return np.random.rand(*shape).astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="OpenVINO 基础示例：加载并推理模型")
    parser.add_argument("--model", "-m", required=True, help="模型路径 (.onnx 或 .xml)")
    parser.add_argument("--device", "-d", default="CPU", help="设备，例如 CPU 或 GPU")
    parser.add_argument("--warmup", type=int, default=2, help="预热次数")
    parser.add_argument("--iterations", "-i", type=int, default=10, help="测量次数")
    args = parser.parse_args()

    try:
        from openvino.runtime import Core
    except Exception as e:
        raise RuntimeError("无法导入 openvino.runtime，请先安装 openvino/openvino-dev。错误: {}".format(e))

    core = Core()
    model = core.read_model(model=args.model)
    compiled = core.compile_model(model, args.device)

    inp = compiled.input(0)
    input_shape = normalize_shape(tuple(getattr(inp, "shape", ())))
    print("模型输入形状 (采样):", input_shape)

    sample = make_random_input(input_shape)

    # 预热
    for _ in range(args.warmup):
        _ = compiled([sample])

    # 多次测量
    times = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        out = compiled([sample])
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times_sorted = sorted(times)
    avg = sum(times) / len(times)
    median = times_sorted[len(times_sorted)//2]
    p95 = times_sorted[min(int(len(times_sorted)*0.95), len(times_sorted)-1)]

    # 输出摘要
    print("重复次数:", args.iterations, "预热:", args.warmup)
    print("延迟 times (s):", times)
    print(f"avg={avg:.6f}s median={median:.6f}s p95={p95:.6f}s throughput={1.0/avg:.2f}fps")
    # 若需要查看输出内容/shape
    try:
        if isinstance(out, dict):
            print("输出键:", list(out.keys()))
        else:
            o = out[0] if isinstance(out, (list, tuple)) else out
            print("输出类型:", type(o), "shape:", getattr(o, "shape", None))
    except Exception:
        pass

if __name__ == "__main__":
    main()