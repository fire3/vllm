import { CodeTabs } from "/home/fire3/SRC/vllm/learn/doc/node_modules/.pnpm/@vuepress+plugin-markdown-tab@2.0.0-rc.132_@vuepress+bundler-vite@2.0.0-rc.31_@types+no_f3ee1c360f41fd7ca1328a1ce47d23c1/node_modules/@vuepress/plugin-markdown-tab/dist/client/components/CodeTabs.js";
import { Tabs } from "/home/fire3/SRC/vllm/learn/doc/node_modules/.pnpm/@vuepress+plugin-markdown-tab@2.0.0-rc.132_@vuepress+bundler-vite@2.0.0-rc.31_@types+no_f3ee1c360f41fd7ca1328a1ce47d23c1/node_modules/@vuepress/plugin-markdown-tab/dist/client/components/Tabs.js";

export default {
  enhance: ({ app }) => {
    app.component("CodeTabs", CodeTabs);
    app.component("Tabs", Tabs);
  },
};
