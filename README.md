# reranker_npu

# 安装依赖
- 下载最近release.zip
- 安装ryzen-ai-1.6.0
- conda activate ryzen-ai-1.6.0
- pip install transformers torch onnxscipt

# 快速开始

#### 运行

```bash
python test.py
```

#### 结果
```json
{
	"results": [{
		"candidate": "深いテーマを持ちながらも、観る人の心を揺さぶる名作。登場人物の心情描写が秀逸で、ラストは涙なしでは見られない。",
		"score": 0.5457699298858643
	}, {
		"candidate": "どうにもリアリティに欠ける展開が気になった。もっと深みのある人間ドラマが見たかった。",
		"score": 0.5414088368415833
	}, {
		"candidate": "アクションシーンが楽しすぎる。見ていて飽きない。ストーリーはシンプルだが、それが逆に良い。",
		"score": 0.5346125364303589
	}, {
		"candidate": "重要なメッセージ性は評価で きるが、暗い話が続くので気分が落ち込んでしまった。もう少し明るい要素があればよかった。",
		"score": 0.5147662162780762
	}]
}
```

# 从头开始

```bash
# 下载导出onnx
python export2onnx.py
# 量化模型
python quantize.py
# 执行测试
python test.py
```
