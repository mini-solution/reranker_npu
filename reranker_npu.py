import os
import onnxruntime
from transformers import AutoTokenizer
import numpy as np

class reranker:
    def __init__(self,query,candidates):
        self.query = query
        self.candidates = candidates
    def predit(self):
        onnx_dir = "./onnx_static"
        tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
        install_dir = os.environ['RYZEN_AI_INSTALLATION_PATH']
        config_file = "vaiep_config.json"
        # print(install_dir)
        cache_dir = os.path.abspath('cache')
        cache_key   = "reranker"

        options = onnxruntime.SessionOptions()
        options.log_severity_level = 0   # 0 = VERBOSE
        print(onnxruntime.get_available_providers())
        session = onnxruntime.InferenceSession("onnx_static/quantize.onnx",    
            providers=[
                'VitisAIExecutionProvider',
                # "DmlExecutionProvider",
                # "CPUExecutionProvider",
            ],
            sess_options=options,
            provider_options = [{
                        "config_file": config_file,
                        "cache_dir": cache_dir,
                        "cache_key": cache_key,
                        "enable_cache_file_io_in_mem":0,
                        # 'xclbin': xclbin_file
                    }]
        )

        # for i in (0,20):
            # start = datetime.now()
        scores = []
        for candidate in self.candidates:
            inputs = tokenizer(self.query, candidate, return_tensors="np", padding="max_length", truncation=True,max_length=128)
            # ONNX 模型要求输入是 int64
            onnx_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
            # 执行推理
            logit = session.run(None, onnx_inputs)[0]
            # 安全地计算概率（sigmoid）
            prob = 1 / (1 + np.exp(-logit))
            scores.append(float(prob.squeeze()))
        sorted_candidates = [x for _, x in sorted(zip(scores, self.candidates), reverse=True)]
        return ({
            "scores":scores,
            "candidates": sorted_candidates,
        })
