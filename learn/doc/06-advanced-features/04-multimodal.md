# 多模态

## 文件位置

`vllm/multimodal/` 目录

## 处理流程

多模态请求的处理链路：

```
多模态输入 (URL / base64 / 本地文件)
    │
    ├── parse.py: 解析输入来源，读取数据
    │
    ├── inputs.py: 构建 MultiModalFeatureSpec
    │
    ├── processing/: 预处理管道
    │   ├── 图像：解码 → resize → normalize → tensor
    │   ├── 音频：解码 → 重采样 → 特征提取 → tensor
    │   └── 视频：解码 → 抽帧 → 逐帧图像处理
    │
    ├── cache.py: 缓存已编码的多模态数据（避免重复编码）
    │
    └── 模型侧的 embed_multimodal() 处理
```

## 支持模态

| 模态 | 处理文件 | 编码器 | 说明 |
|------|---------|--------|------|
| 图像 | `image.py` | CLIP / SigLIP / InternViT 等 | 支持多分辨率、动态分辨率 |
| 音频 | `audio.py` (14KB) | Whisper / 其他 | 支持多种采样率 |
| 视频 | `video.py` (75KB) | 同上图像编码器 + 时序处理 | 支持关键帧提取、可变帧率 |

## 多模态模型注册

`multimodal/registry.py` 注册支持多模态的模型。

模型的 `SupportsMultiModal` Protocol 要求实现：

```python
class SupportsMultiModal(Protocol):
    def get_placeholder_str(self) -> str:
        """返回多模态占位符（如 <image>）"""
    def embed_multimodal(self, data) -> MultiModalEmbeddings:
        """编码多模态数据为 embedding"""
    def configure_mm_token_handling(self, ...):
        """配置多模态 token 的处理方式"""
```
