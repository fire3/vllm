import { GitContributors } from "/home/fire3/SRC/vllm/learn/doc/node_modules/.pnpm/@vuepress+plugin-git@2.0.0-rc.132_@vuepress+bundler-vite@2.0.0-rc.31_@types+node@26.1.1_af2e071db8bdb006ba1e14dd384c9d31/node_modules/@vuepress/plugin-git/dist/client/components/GitContributors.js";
import { GitChangelog } from "/home/fire3/SRC/vllm/learn/doc/node_modules/.pnpm/@vuepress+plugin-git@2.0.0-rc.132_@vuepress+bundler-vite@2.0.0-rc.31_@types+node@26.1.1_af2e071db8bdb006ba1e14dd384c9d31/node_modules/@vuepress/plugin-git/dist/client/components/GitChangelog.js";

export default {
  enhance: ({ app }) => {
    app.component("GitContributors", GitContributors);
    app.component("GitChangelog", GitChangelog);
  },
};
