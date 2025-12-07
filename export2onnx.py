from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

save_dir = "onnx_model/"
os.makedirs(save_dir, exist_ok=True)
model_name = "maidalun1020/bce-reranker-base_v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()
# 准备输入示例
# text_a = "これは日本語のテキストです。"
# text_b = "これは別の文です。"
# inputs = tokenizer(text_a, text_b, return_tensors="pt", padding=True, truncation=True, max_length=128)

batch_size = 1
seq_len = 128  # 想要固定的长度
dummy_input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
torch.onnx.export(
    model,
    # (inputs["input_ids"], inputs["attention_mask"]),
    (dummy_input_ids, dummy_attention_mask),
    f"{save_dir}model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    # dynamic_axes={
    #     "input_ids": {0: "batch", 1: "sequence"},
    #     "attention_mask": {0: "batch", 1: "sequence"},
    #     "last_hidden_state": {0: "batch", 1: "sequence"},
    # },
    dynamic_axes=None, 
    opset_version=17,
)

# save_dir = "onnx_static/"
# os.makedirs(save_dir, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_dir)

# fixed shape
import subprocess
params_to_fix = {
    "batch": 1,        # 有时是 "batch_size"
    "sequence": 128,    # 也可能是 "sequence_length"
}
input_model_path = 'onnx_model/model.onnx'
output_model_path = 'onnx_model/model_fixed.onnx'
command_base = ["python", "-m", "onnxruntime.tools.make_dynamic_shape_fixed"]
for param, value in params_to_fix.items():
    command = command_base + [input_model_path, output_model_path, "--dim_param", str(param), "--dim_value", str(value)]
    subprocess.run(command)
    input_model_path = output_model_path
print("Static conversion complete.")


import onnx
from onnxsim import simplify
model = onnx.load("onnx_model/model_fixed.onnx")
model_simp, check = simplify(model)
onnx.save(model_simp, "onnx_model/model_sim.onnx")
if check:
    print("✅ 模型简化成功并通过验证")
else:
    print("⚠️ 模型简化验证失败，请检查输入形状或动态轴定义")


from onnx import shape_inference
input_model_path = 'onnx_model/model_sim.onnx'
output_model_path = 'onnx_model/model_shaped.onnx'
model = onnx.load(input_model_path)
inferred_model = shape_inference.infer_shapes(model, data_prop=True)
onnx.save_model(inferred_model, output_model_path)
print(f"Shape inference complete. Overwritten: {output_model_path}")
