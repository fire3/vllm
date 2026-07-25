<template><div><h2 id="项目概述" tabindex="-1"><a class="header-anchor" href="#项目概述"><span>项目概述</span></a></h2>
<p><a href="https://github.com/vllm-project/vllm" target="_blank" rel="noopener noreferrer">vLLM<ExternalLinkIcon/></a> 是一个高性能的大语言模型（LLM）推理引擎，由加州大学伯克利分校开发并开源。其核心创新包括：</p>
<ul>
<li><strong>PagedAttention</strong>：受操作系统分页思想启发的 KV Cache 管理算法，近乎零碎片的内存利用</li>
<li><strong>Continuous Batching</strong>：连续批处理，Prefill 和 Decode 阶段在同一个 batch 中混合执行</li>
<li><strong>极致推理性能</strong>：通过 CUDA Graph、算子融合、量化等优化达到业界领先的推理吞吐量</li>
<li><strong>广泛模型支持</strong>：100+ 模型架构（LLaMA、Qwen、DeepSeek、Gemma、Mistral、Phi 等）</li>
<li><strong>多模态</strong>：文本 + 图像 + 音频 + 视频的统一推理</li>
<li><strong>分布式并行</strong>：张量并行、流水线并行、专家并行等全面的分布式推理能力</li>
</ul>
<p>本学习项目按 <strong>七个阶段</strong> 渐进式深入源码，从系统底座到上层服务，从单 GPU 推理到分布式部署，适合有一定 PyTorch 和 Transformer 基础的开发者。</p>
<h2 id="如何使用" tabindex="-1"><a class="header-anchor" href="#如何使用"><span>如何使用</span></a></h2>
<div class="language-bash line-numbers-mode" data-highlighter="prismjs" data-ext="sh"><pre v-pre><code class="language-bash"><span class="line"><span class="token comment"># 安装依赖后本地运行 VuePress</span></span>
<span class="line"><span class="token builtin class-name">cd</span> learn/doc</span>
<span class="line"><span class="token function">pnpm</span> <span class="token function">install</span></span>
<span class="line"><span class="token function">pnpm</span> run dev</span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>每个阶段的分析文档都包含：</p>
<ul>
<li>要阅读的具体代码文件路径（含文件大小参考）</li>
<li>关键概念解释和架构流程图</li>
<li>学习产出清单与思考题</li>
</ul>
</div></template>


