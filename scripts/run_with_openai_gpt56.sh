#!/usr/bin/env bash
# 兼容旧入口：统一委托给多提供方启动器及同一受限密钥文件。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$script_dir/run_with_llm_provider.sh" openai "$@"
