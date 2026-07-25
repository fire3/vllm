export const redirects = JSON.parse("{}")

export const routes = Object.fromEntries([
  ["/", { loader: () => import(/* webpackChunkName: "index.html" */"/home/fire3/SRC/vllm/learn/doc/README.md"), meta: {"title":"首页"} }],
  ["/01-overview/01-config-system.html", { loader: () => import(/* webpackChunkName: "01-overview_01-config-system.html" */"/home/fire3/SRC/vllm/learn/doc/01-overview/01-config-system.md"), meta: {"title":"配置系统"} }],
  ["/01-overview/02-logging-monitoring.html", { loader: () => import(/* webpackChunkName: "01-overview_02-logging-monitoring.html" */"/home/fire3/SRC/vllm/learn/doc/01-overview/02-logging-monitoring.md"), meta: {"title":"日志、追踪与监控"} }],
  ["/01-overview/03-build-system.html", { loader: () => import(/* webpackChunkName: "01-overview_03-build-system.html" */"/home/fire3/SRC/vllm/learn/doc/01-overview/03-build-system.md"), meta: {"title":"编译与构建系统"} }],
  ["/01-overview/", { loader: () => import(/* webpackChunkName: "01-overview_index.html" */"/home/fire3/SRC/vllm/learn/doc/01-overview/README.md"), meta: {"title":"阶段一：系统底座"} }],
  ["/404.html", { loader: () => import(/* webpackChunkName: "404.html" */"/home/fire3/SRC/vllm/learn/doc/.vuepress/.temp/pages/404.html.vue"), meta: {"title":""} }],
]);

if (import.meta.webpackHot) {
  import.meta.webpackHot.accept()
  __VUE_HMR_RUNTIME__.updateRoutes?.(routes)
  __VUE_HMR_RUNTIME__.updateRedirects?.(redirects)
}

if (import.meta.hot) {
  import.meta.hot.accept((m) => {
    __VUE_HMR_RUNTIME__.updateRoutes?.(m.routes)
    __VUE_HMR_RUNTIME__.updateRedirects?.(m.redirects)
  })
}
