#!/usr/bin/env bash
set -Eeuo pipefail

HF_TOKEN="${1:-${HF_TOKEN:-}}"
if [[ -z "${HF_TOKEN}" ]]; then
  echo "❌  Supply your Hugging Face token as a CLI arg or HF_TOKEN env var."
  exit 1
fi

if ! grep -q "lerna.tools.corp.linkedin.com" "${HOME}/.bashrc"; then
cat <<'EOF' >> "${HOME}/.bashrc"

# ---- internal proxy exceptions ----
export NO_PROXY="lerna.tools.corp.linkedin.com,\
artifactory.corp.linkedin.com,\
mlflow.grid1.ard.grid.linkedin.com,\
mlflow.grid1.ard.grid.linkedin.com"
EOF
fi

# also expose the variable for *this* run
export NO_PROXY="lerna.tools.corp.linkedin.com,artifactory.corp.linkedin.com,mlflow.grid1.ard.grid.linkedin.com,mlflow.grid1.ard.grid.linkedin.com"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt   # make sure the filename is spelled correctly

huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential   # --token flag avoids prompts:contentReference[oaicite:0]{index=0}

echo -e "\n✅  Environment ready.  Open a new shell (or 'source ~/.bashrc') for NO_PROXY to be available interactively."
