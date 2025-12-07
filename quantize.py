from transformers import AutoTokenizer
from onnxruntime.quantization import CalibrationDataReader
import numpy as np
import onnx

class TextCalibrationReader(CalibrationDataReader):
    def __init__(self, input_ids_name="input_ids", attn_mask_name="attention_mask", num_samples=10):
        self.input_ids_name = input_ids_name
        self.attn_mask_name = attn_mask_name
        self.tokenizer = AutoTokenizer.from_pretrained("onnx_model")  # 确保是模型文件夹
        texts = [
            "これは感動的な映画です。",
            "面白いストーリーとキャラクター。",
            "映像が美しい作品。",
            "内容が少し退屈だった。",
            "音楽がとても印象的だった。"
        ]
        self.enum_data = iter([
            self._encode_text(t)
            for t in texts[:num_samples]
        ])

    def _encode_text(self, t):
        encoded = self.tokenizer(
            t,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=64
        )
        return {
            self.input_ids_name: encoded["input_ids"].astype(np.int64),
            self.attn_mask_name: encoded["attention_mask"].astype(np.int64),
        }

    def get_next(self):
        return next(self.enum_data, None)

from onnxruntime.quantization.quant_utils import QuantFormat, QuantType
from quark.onnx.quantization.config import QConfig
from quark.onnx import ModelQuantizer

# reader = TextCalibrationReader("input_ids", "attention_mask")
quant_config = QConfig.get_default_config("BF16")
quant_config.global_quant_config.extra_options["BF16QDQToCast"] = True
quant_config.global_quant_config.extra_options["UseRandomData"] = True
quantizer = ModelQuantizer(quant_config)
quantizer.quantize_model(
    model_input="onnx_model/model_shaped.onnx",
    model_output="onnx_static/quantize.onnx",
    calibration_data_path=None
    # calibration_data_reader=reader
)