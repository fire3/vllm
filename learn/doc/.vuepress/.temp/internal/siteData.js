export const siteData = JSON.parse("{\"base\":\"/\",\"lang\":\"zh-CN\",\"title\":\"vLLM 源码分析\",\"description\":\"逐步深入 vLLM 高性能 LLM 推理框架源码\",\"head\":[],\"locales\":{\"/\":{\"lang\":\"zh-CN\",\"title\":\"vLLM 源码分析\",\"description\":\"逐步深入 vLLM 高性能 LLM 推理框架源码\"}}}")

if (import.meta.webpackHot) {
  import.meta.webpackHot.accept()
  __VUE_HMR_RUNTIME__.updateSiteData?.(siteData)
}

if (import.meta.hot) {
  import.meta.hot.accept((m) => {
    __VUE_HMR_RUNTIME__.updateSiteData?.(m.siteData)
  })
}
