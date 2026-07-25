import { defineUserConfig } from "vuepress";
import { viteBundler } from "@vuepress/bundler-vite";
import { defaultTheme } from "@vuepress/theme-default";
import { searchPlugin } from "@vuepress/plugin-search";

export default defineUserConfig({
  lang: "zh-CN",
  title: "vLLM 源码分析",
  description: "逐步深入 vLLM 高性能 LLM 推理框架源码",

  bundler: viteBundler(),

  theme: defaultTheme({
    logo: null,
    repo: "https://github.com/vllm-project/vllm",
    sidebar: [
      {
        text: "总览",
        link: "/",
      },
      {
        text: "阶段一：系统底座",
        link: "/01-overview/",
        children: [
          "/01-overview/01-config-system.md",
          "/01-overview/02-logging-monitoring.md",
          "/01-overview/03-build-system.md",
        ],
      },
      {
        text: "阶段二：核心推理管线",
        link: "/02-inference-pipeline/",
        children: [
          "/02-inference-pipeline/01-request-lifecycle.md",
          "/02-inference-pipeline/02-kv-cache-system.md",
          "/02-inference-pipeline/03-scheduler.md",
          "/02-inference-pipeline/04-attention-backend.md",
          "/02-inference-pipeline/05-sampler.md",
          "/02-inference-pipeline/06-worker-model-runner.md",
          "/02-inference-pipeline/07-engine-layer.md",
        ],
      },
      {
        text: "阶段三：模型层",
        link: "/03-model-layer/",
        children: [
          "/03-model-layer/01-model-interfaces.md",
          "/03-model-layer/02-neural-layers.md",
          "/03-model-layer/03-model-implementations.md",
          "/03-model-layer/04-model-loading.md",
        ],
      },
      {
        text: "阶段四：服务与入口层",
        link: "/04-serving-layer/",
        children: [
          "/04-serving-layer/01-offline-api.md",
          "/04-serving-layer/02-openai-api.md",
          "/04-serving-layer/03-http-server.md",
          "/04-serving-layer/04-other-entrypoints.md",
          "/04-serving-layer/05-request-flow.md",
        ],
      },
      {
        text: "阶段五：分布式与并行",
        link: "/05-distributed-parallel/",
        children: [
          "/05-distributed-parallel/01-parallel-strategies.md",
          "/05-distributed-parallel/02-communication.md",
          "/05-distributed-parallel/03-kv-weight-transfer.md",
          "/05-distributed-parallel/04-coordination.md",
        ],
      },
      {
        text: "阶段六：高级特性",
        link: "/06-advanced-features/",
        children: [
          "/06-advanced-features/01-quantization.md",
          "/06-advanced-features/02-lora.md",
          "/06-advanced-features/03-speculative-decoding.md",
          "/06-advanced-features/04-multimodal.md",
          "/06-advanced-features/05-structured-output.md",
          "/06-advanced-features/06-kv-offload.md",
        ],
      },
      {
        text: "阶段七：工具链与工程",
        link: "/07-toolchain-engineering/",
        children: [
          "/07-toolchain-engineering/01-cuda-kernels.md",
          "/07-toolchain-engineering/02-performance-optimization.md",
          "/07-toolchain-engineering/03-testing.md",
          "/07-toolchain-engineering/04-engineering-config.md",
        ],
      },
    ],
  }),

  plugins: [searchPlugin({})],
});
