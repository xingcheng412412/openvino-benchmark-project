# OpenVINO 基础使用说明（含环境安装步骤）

本文件说明如何在 Linux（如 Ubuntu / VirtualBox）上为本项目创建 Python 虚拟环境并安装 OpenVINO，以及如何验证与运行示例脚本 `src/openvino_basic.py`。

重要说明
- 建议在项目目录下使用虚拟环境（避免污染系统 Python）。
- 若要使用 GPU，请先安装并配置对应硬件驱动与 OpenVINO GPU 插件（本文只给出 CPU 常规安装步骤）。

---

## 前置依赖（系统层面）
在开始前请确保系统已安装 Python3 与 venv 支持以及常见构建工具：

```bash
# Ubuntu / Debian 示例
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential wget
```

---

## 在项目下创建并激活虚拟环境（推荐）
在项目根目录下执行：

```bash
cd /home/sun/openvino-benchmark-project
python3 -m venv .venv
# 激活虚拟环境
source .venv/bin/activate
# 升级 pip
python -m pip install --upgrade pip
```

> 注：在 VirtualBox/虚拟机中如缺少 `python3-venv` 请先用 apt 安装（见上）。

---

## 安装 OpenVINO（CPU 基本、推荐使用 openvino-dev）
推荐安装 `openvino-dev[all]`（包含 runtime 与常用拓展）。安装命令：

```bash
# 已激活虚拟环境时运行
python -m pip install --upgrade "openvino-dev[all]" numpy
```

如只需运行已编译好的模型而不需要开发包，可尝试：
```bash
python -m pip install --upgrade openvino numpy
```

如果需要 ONNX baseline（onnxruntime）：
```bash
python -m pip install --upgrade onnxruntime
```

---

## 验证安装是否成功
在虚拟环境中执行：

```bash
# 验证 openvino runtime 可用
python -c "from openvino.runtime import Core; print('openvino.runtime OK, Core=', Core)"

# 验证 onnxruntime（若已安装）
python -c "import onnxruntime as ort; print('onnxruntime OK, version=', ort.__version__)"
```

若出现 `ModuleNotFoundError: No module named 'openvino.runtime'`，请确认使用的是虚拟环境里的 python 并已安装 `openvino-dev[all]`。

---

## 运行示例脚本
使用项目虚拟环境的 python 运行基础示例：

```bash
# 激活 venv（如未激活）
cd /home/sun/openvino-benchmark-project
source .venv/bin/activate

# 运行（替换为你实际模型路径）
python src/openvino_basic.py --model src/best_openvino_model/best.xml --device CPU --iterations 20 --warmup 3
```

或者直接使用绝对 venv python：
```bash
/home/sun/openvino-benchmark-project/.venv/bin/python /home/sun/openvino-benchmark-project/src/openvino_basic.py --model /home/sun/openvino-benchmark-project/src/best_openvino_model/best.xml --device CPU --iterations 20 --warmup 3
```

脚本会打印模型输入 shape、每次推理延迟、avg/median/p95 与吞吐（fps）。

---

## 常见问题与排查
- 无法导入 openvino.runtime：
  - 确保激活了虚拟环境并在该环境内安装了 `openvino-dev[all]`。
  - 使用 `which python` / `python -m pip show openvino` 确认路径一致。

- 模型无法打开：
  - 确认模型路径正确；若是 IR 格式（.xml），确保同目录下存在对应的 .bin 文件。
  - 对于 ONNX 文件，确认文件完整可读。

- 内存/性能问题：
  - 在虚拟机中，分配足够内存/CPU 核心会显著影响结果。若测得延迟很高，检查 VM 资源设置。
  - 可通过增加 `--iterations` 与 `--warmup` 获取更稳定数值。

- GPU 使用：
  - 需安装合适 GPU 驱动与 OpenVINO GPU 插件，且在 `compile_model` 时选择对应 device（例 `GPU`）。具体请参考 Intel/OpenVINO 官方文档。

---

## 进一步建议
- 若需对比不同实现（ONNXRuntime / OpenVINO / Numpy），可分别在同一虚拟环境中安装所需包并保存结果为 JSON 便于对比。
- 如需自动化运行多批次/多设备基准，可扩展 `openvino_basic.py` 增加 `--batch` 与 `--perf_hint` 参数。

---


