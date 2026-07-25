export const SEARCH_INDEX = [
  {
    "title": "首页",
    "headers": [
      {
        "level": 2,
        "title": "项目概述",
        "slug": "项目概述",
        "link": "#项目概述",
        "children": []
      },
      {
        "level": 2,
        "title": "如何使用",
        "slug": "如何使用",
        "link": "#如何使用",
        "children": []
      }
    ],
    "path": "/",
    "pathLocale": "/",
    "extraFields": []
  },
  {
    "title": "配置系统",
    "headers": [
      {
        "level": 2,
        "title": "为什么需要双层配置？",
        "slug": "为什么需要双层配置",
        "link": "#为什么需要双层配置",
        "children": []
      },
      {
        "level": 2,
        "title": "1. 环境变量系统（vllm/envs.py）",
        "slug": "_1-环境变量系统-vllm-envs-py",
        "link": "#_1-环境变量系统-vllm-envs-py",
        "children": [
          {
            "level": 3,
            "title": "文件位置",
            "slug": "文件位置",
            "link": "#文件位置",
            "children": []
          },
          {
            "level": 3,
            "title": "设计模式",
            "slug": "设计模式",
            "link": "#设计模式",
            "children": []
          },
          {
            "level": 3,
            "title": "按功能分类的关键环境变量",
            "slug": "按功能分类的关键环境变量",
            "link": "#按功能分类的关键环境变量",
            "children": []
          },
          {
            "level": 3,
            "title": "阅读要点",
            "slug": "阅读要点",
            "link": "#阅读要点",
            "children": []
          }
        ]
      },
      {
        "level": 2,
        "title": "2. 引擎参数系统（vllm/engine/arg_utils.py）",
        "slug": "_2-引擎参数系统-vllm-engine-arg-utils-py",
        "link": "#_2-引擎参数系统-vllm-engine-arg-utils-py",
        "children": [
          {
            "level": 3,
            "title": "文件位置",
            "slug": "文件位置-1",
            "link": "#文件位置-1",
            "children": []
          },
          {
            "level": 3,
            "title": "EngineArgs 类",
            "slug": "engineargs-类",
            "link": "#engineargs-类",
            "children": []
          },
          {
            "level": 3,
            "title": "配置生产管线",
            "slug": "配置生产管线",
            "link": "#配置生产管线",
            "children": []
          },
          {
            "level": 3,
            "title": "关键配置对象",
            "slug": "关键配置对象",
            "link": "#关键配置对象",
            "children": []
          },
          {
            "level": 3,
            "title": "CLI 入口",
            "slug": "cli-入口",
            "link": "#cli-入口",
            "children": []
          }
        ]
      },
      {
        "level": 2,
        "title": "3. 平台抽象层（vllm/platforms/）",
        "slug": "_3-平台抽象层-vllm-platforms",
        "link": "#_3-平台抽象层-vllm-platforms",
        "children": [
          {
            "level": 3,
            "title": "核心概念",
            "slug": "核心概念",
            "link": "#核心概念",
            "children": []
          },
          {
            "level": 3,
            "title": "目录结构",
            "slug": "目录结构",
            "link": "#目录结构",
            "children": []
          },
          {
            "level": 3,
            "title": "自动检测逻辑",
            "slug": "自动检测逻辑",
            "link": "#自动检测逻辑",
            "children": []
          },
          {
            "level": 3,
            "title": "Platform 基类的关键方法",
            "slug": "platform-基类的关键方法",
            "link": "#platform-基类的关键方法",
            "children": []
          }
        ]
      },
      {
        "level": 2,
        "title": "学习产出清单",
        "slug": "学习产出清单",
        "link": "#学习产出清单",
        "children": []
      },
      {
        "level": 2,
        "title": "思考题",
        "slug": "思考题",
        "link": "#思考题",
        "children": []
      },
      {
        "level": 2,
        "title": "下一步",
        "slug": "下一步",
        "link": "#下一步",
        "children": []
      }
    ],
    "path": "/01-overview/01-config-system.html",
    "pathLocale": "/",
    "extraFields": []
  },
  {
    "title": "日志、追踪与监控",
    "headers": [
      {
        "level": 2,
        "title": "1. 分布式日志系统（vllm/logger.py）",
        "slug": "_1-分布式日志系统-vllm-logger-py",
        "link": "#_1-分布式日志系统-vllm-logger-py",
        "children": [
          {
            "level": 3,
            "title": "文件位置",
            "slug": "文件位置",
            "link": "#文件位置",
            "children": []
          },
          {
            "level": 3,
            "title": "核心问题",
            "slug": "核心问题",
            "link": "#核心问题",
            "children": []
          },
          {
            "level": 3,
            "title": "实现机制",
            "slug": "实现机制",
            "link": "#实现机制",
            "children": []
          },
          {
            "level": 3,
            "title": "使用方式",
            "slug": "使用方式",
            "link": "#使用方式",
            "children": []
          },
          {
            "level": 3,
            "title": "关键要点",
            "slug": "关键要点",
            "link": "#关键要点",
            "children": []
          }
        ]
      },
      {
        "level": 2,
        "title": "2. OpenTelemetry 分布式追踪（vllm/tracing/）",
        "slug": "_2-opentelemetry-分布式追踪-vllm-tracing",
        "link": "#_2-opentelemetry-分布式追踪-vllm-tracing",
        "children": [
          {
            "level": 3,
            "title": "文件位置",
            "slug": "文件位置-1",
            "link": "#文件位置-1",
            "children": []
          },
          {
            "level": 3,
            "title": "什么是分布式追踪？",
            "slug": "什么是分布式追踪",
            "link": "#什么是分布式追踪",
            "children": []
          },
          {
            "level": 3,
            "title": "OpenTelemetry 集成",
            "slug": "opentelemetry-集成",
            "link": "#opentelemetry-集成",
            "children": []
          },
          {
            "level": 3,
            "title": "关键追踪阶段",
            "slug": "关键追踪阶段",
            "link": "#关键追踪阶段",
            "children": []
          },
          {
            "level": 3,
            "title": "启用方式",
            "slug": "启用方式",
            "link": "#启用方式",
            "children": []
          }
        ]
      },
      {
        "level": 2,
        "title": "3. Prometheus 指标（vllm/v1/metrics/）",
        "slug": "_3-prometheus-指标-vllm-v1-metrics",
        "link": "#_3-prometheus-指标-vllm-v1-metrics",
        "children": [
          {
            "level": 3,
            "title": "文件位置",
            "slug": "文件位置-2",
            "link": "#文件位置-2",
            "children": []
          },
          {
            "level": 3,
            "title": "暴露的指标",
            "slug": "暴露的指标",
            "link": "#暴露的指标",
            "children": []
          },
          {
            "level": 3,
            "title": "使用方式",
            "slug": "使用方式-1",
            "link": "#使用方式-1",
            "children": []
          }
        ]
      },
      {
        "level": 2,
        "title": "4. 使用统计（vllm/usage/）",
        "slug": "_4-使用统计-vllm-usage",
        "link": "#_4-使用统计-vllm-usage",
        "children": [
          {
            "level": 3,
            "title": "核心目的",
            "slug": "核心目的",
            "link": "#核心目的",
            "children": []
          },
          {
            "level": 3,
            "title": "收集的内容",
            "slug": "收集的内容",
            "link": "#收集的内容",
            "children": []
          },
          {
            "level": 3,
            "title": "控制方式",
            "slug": "控制方式",
            "link": "#控制方式",
            "children": []
          }
        ]
      },
      {
        "level": 2,
        "title": "学习产出清单",
        "slug": "学习产出清单",
        "link": "#学习产出清单",
        "children": []
      },
      {
        "level": 2,
        "title": "思考题",
        "slug": "思考题",
        "link": "#思考题",
        "children": []
      },
      {
        "level": 2,
        "title": "下一步",
        "slug": "下一步",
        "link": "#下一步",
        "children": []
      }
    ],
    "path": "/01-overview/02-logging-monitoring.html",
    "pathLocale": "/",
    "extraFields": []
  },
  {
    "title": "编译与构建系统",
    "headers": [
      {
        "level": 2,
        "title": "1. 总体架构",
        "slug": "_1-总体架构",
        "link": "#_1-总体架构",
        "children": []
      },
      {
        "level": 2,
        "title": "2. Python 包构建（setup.py）",
        "slug": "_2-python-包构建-setup-py",
        "link": "#_2-python-包构建-setup-py",
        "children": [
          {
            "level": 3,
            "title": "文件位置",
            "slug": "文件位置",
            "link": "#文件位置",
            "children": []
          },
          {
            "level": 3,
            "title": "扩展注册",
            "slug": "扩展注册",
            "link": "#扩展注册",
            "children": []
          },
          {
            "level": 3,
            "title": "VLLM_USE_PRECOMPILED",
            "slug": "vllm-use-precompiled",
            "link": "#vllm-use-precompiled",
            "children": []
          },
          {
            "level": 3,
            "title": "条件编译",
            "slug": "条件编译",
            "link": "#条件编译",
            "children": []
          }
        ]
      },
      {
        "level": 2,
        "title": "3. CMake 构建层（CMakeLists.txt）",
        "slug": "_3-cmake-构建层-cmakelists-txt",
        "link": "#_3-cmake-构建层-cmakelists-txt",
        "children": [
          {
            "level": 3,
            "title": "文件位置",
            "slug": "文件位置-1",
            "link": "#文件位置-1",
            "children": []
          },
          {
            "level": 3,
            "title": "为什么需要 CMake？",
            "slug": "为什么需要-cmake",
            "link": "#为什么需要-cmake",
            "children": []
          },
          {
            "level": 3,
            "title": "关键 CMake 结构",
            "slug": "关键-cmake-结构",
            "link": "#关键-cmake-结构",
            "children": []
          },
          {
            "level": 3,
            "title": "子目录结构",
            "slug": "子目录结构",
            "link": "#子目录结构",
            "children": []
          }
        ]
      },
      {
        "level": 2,
        "title": "4. Python↔C++ 绑定（torch_bindings.cpp）",
        "slug": "_4-python↔c-绑定-torch-bindings-cpp",
        "link": "#_4-python↔c-绑定-torch-bindings-cpp",
        "children": [
          {
            "level": 3,
            "title": "文件位置",
            "slug": "文件位置-2",
            "link": "#文件位置-2",
            "children": []
          },
          {
            "level": 3,
            "title": "绑定机制",
            "slug": "绑定机制",
            "link": "#绑定机制",
            "children": []
          },
          {
            "level": 3,
            "title": "_custom_ops.py 的角色",
            "slug": "custom-ops-py-的角色",
            "link": "#custom-ops-py-的角色",
            "children": []
          },
          {
            "level": 3,
            "title": "另一个封装：_aiter_ops.py",
            "slug": "另一个封装-aiter-ops-py",
            "link": "#另一个封装-aiter-ops-py",
            "children": []
          }
        ]
      },
      {
        "level": 2,
        "title": "5. Rust 组件",
        "slug": "_5-rust-组件",
        "link": "#_5-rust-组件",
        "children": [
          {
            "level": 3,
            "title": "文件位置",
            "slug": "文件位置-3",
            "link": "#文件位置-3",
            "children": []
          },
          {
            "level": 3,
            "title": "为什么需要 Rust？",
            "slug": "为什么需要-rust",
            "link": "#为什么需要-rust",
            "children": []
          },
          {
            "level": 3,
            "title": "构建方式",
            "slug": "构建方式",
            "link": "#构建方式",
            "children": []
          }
        ]
      },
      {
        "level": 2,
        "title": "6. 依赖管理（requirements/）",
        "slug": "_6-依赖管理-requirements",
        "link": "#_6-依赖管理-requirements",
        "children": [
          {
            "level": 3,
            "title": "目录结构",
            "slug": "目录结构",
            "link": "#目录结构",
            "children": []
          },
          {
            "level": 3,
            "title": "PyTorch 后端的依赖选择",
            "slug": "pytorch-后端的依赖选择",
            "link": "#pytorch-后端的依赖选择",
            "children": []
          }
        ]
      },
      {
        "level": 2,
        "title": "学习产出清单",
        "slug": "学习产出清单",
        "link": "#学习产出清单",
        "children": []
      },
      {
        "level": 2,
        "title": "思考题",
        "slug": "思考题",
        "link": "#思考题",
        "children": []
      },
      {
        "level": 2,
        "title": "全部章节完成",
        "slug": "全部章节完成",
        "link": "#全部章节完成",
        "children": []
      }
    ],
    "path": "/01-overview/03-build-system.html",
    "pathLocale": "/",
    "extraFields": []
  },
  {
    "title": "阶段一：系统底座",
    "headers": [
      {
        "level": 2,
        "title": "章节列表",
        "slug": "章节列表",
        "link": "#章节列表",
        "children": []
      },
      {
        "level": 2,
        "title": "前置知识",
        "slug": "前置知识",
        "link": "#前置知识",
        "children": []
      },
      {
        "level": 2,
        "title": "推荐阅读方法",
        "slug": "推荐阅读方法",
        "link": "#推荐阅读方法",
        "children": []
      }
    ],
    "path": "/01-overview/",
    "pathLocale": "/",
    "extraFields": []
  },
  {
    "title": "",
    "headers": [],
    "path": "/404.html",
    "pathLocale": "/",
    "extraFields": []
  }
]

if (import.meta.webpackHot) {
  import.meta.webpackHot.accept()
  if (__VUE_HMR_RUNTIME__.updateSearchIndex) {
    __VUE_HMR_RUNTIME__.updateSearchIndex(searchIndex)
  }
}

if (import.meta.hot) {
  import.meta.hot.accept(({ searchIndex }) => {
    __VUE_HMR_RUNTIME__.updateSearchIndex(searchIndex)
  })
}
