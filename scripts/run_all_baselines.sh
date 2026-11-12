#!/bin/zsh
# Every baseline x modality, all datasets. Log: experiments/run_baselines.log
export PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH MLFLOW_DISABLE_AGENT_HINT=1
cd "$(dirname "$0")/.."
for m in anomalyclip faprompt adaclip; do
  for mod in chest fundus mri oct; do
    echo "=== $m $mod"
    uv run python scripts/run_baseline.py baselines=$m modality_name=$mod dataset=null 2>&1 \
      | tr '\r' '\n' | grep -E "^(chest|fundus|mri|oct)/|\[skip\]|Error|Traceback"
  done
done
echo DONE
