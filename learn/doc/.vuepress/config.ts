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
    ],
  }),

  plugins: [searchPlugin({})],
});
