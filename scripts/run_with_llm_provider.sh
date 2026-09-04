#!/usr/bin/env bash
# 使用指定提供方启动命令；密钥只从受限的用户配置文件读取。
set -euo pipefail

provider="${1:?用法：$0 <openai|deepseek|dashscope> <命令...>}"
shift
# 调用命令显式传入的模型名优先于持久配置，避免切换 provider 时继承旧模型名。
requested_model_name="${LLM_MODEL_NAME:-}"
env_file="${SCENARIO_DREAMER_LLM_ENV:-$HOME/.config/scenario-dreamer/llm.env}"
if [[ -f "$env_file" ]]; then
  # shellcheck disable=SC1090
  source "$env_file"
fi

case "$provider" in
  openai)
    : "${OPENAI_API_KEY:?请先在 $env_file 设置 OPENAI_API_KEY}"
    export LLM_MODEL_NAME="${requested_model_name:-gpt-5.6}"
    ;;
  gemini)
    : "${GEMINI_API_KEY:?请先在 $env_file 设置 GEMINI_API_KEY}"
    # 免费阶段默认使用 Gemini 2.5 Flash；可在运行前覆盖 LLM_MODEL_NAME。
    export LLM_MODEL_NAME="${requested_model_name:-gemini-2.5-flash}"
    ;;
  qwen|modelscope)
    : "${MODELSCOPE_ACCESS_TOKEN:?请先在 $env_file 设置 MODELSCOPE_ACCESS_TOKEN}"
    # 该模型必须在 ModelScope 页面显示 API Inference 标识后方可免费调用。
    export LLM_PROVIDER="qwen"
    # 当前账户的 API Inference 模型清单不包含 Qwen3.6，使用已验证可列出的 27B 模型。
    export LLM_MODEL_NAME="${requested_model_name:-Qwen/Qwen3.5-27B}"
    ;;
  deepseek)
    : "${DEEPSEEK_API_KEY:?请先在 $env_file 设置 DEEPSEEK_API_KEY}"
    export LLM_MODEL_NAME="${requested_model_name:-deepseek-v4-pro}"
    ;;
  dashscope)
    : "${DASHSCOPE_API_KEY:?请先在 $env_file 设置 DASHSCOPE_API_KEY}"
    export LLM_MODEL_NAME="${requested_model_name:-qwen3.7-plus}"
    ;;
  *)
    echo "不支持的提供方：$provider" >&2
    exit 2
    ;;
esac

export LLM_PROVIDER="$provider"
# 始终把最终模型同步为唯一候选池，防止持久配置遗留的其他服务模型被规划器优先尝试。
export LLM_MODEL_NAMES="$LLM_MODEL_NAME"
exec "$@"
