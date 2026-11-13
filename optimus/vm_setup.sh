#!/usr/bin/env bash
# One-shot setup for a fresh Vast.ai "Ubuntu 22.04 VM" instance, run as root:
#   bash ~/optimus/vm_setup.sh
# Does everything that took an evening by hand on run 1: stops automatic updates,
# installs the NVIDIA container toolkit, tmux, Miniconda, NatureBench and its
# conda environments, and installs OPTIMUS. Safe to run more than once.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APT="apt-get -o DPkg::Lock::Timeout=120"

say() { printf '\n=== %s\n' "$*"; }

wait_for_apt() {
  # Ubuntu's automatic update holds the package lock on a fresh machine and once
  # stayed stuck for 30+ minutes. Wait up to 10 minutes, then stop it cleanly.
  for i in $(seq 1 60); do
    if ! pgrep -f '/usr/bin/unattended-upgrade' >/dev/null 2>&1; then return 0; fi
    echo "   automatic updates are running; waiting ($i/60)..."; sleep 10
  done
  echo "   stopping the automatic update so setup can continue"
  pkill -TERM -f '/usr/bin/unattended-upgrade' || true
  for _ in $(seq 1 60); do pgrep -f '/usr/bin/unattended-upgrade' >/dev/null 2>&1 || break; sleep 5; done
  dpkg --configure -a || true
}

say "1/6 Switching off automatic updates (they lock apt and can replace the GPU driver mid-run)"
systemctl disable --now apt-daily.timer apt-daily-upgrade.timer unattended-upgrades 2>/dev/null || true
wait_for_apt

say "2/6 System packages and NVIDIA container toolkit"
$APT update -y
$APT install -y curl gnupg tmux git wget
if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
  if [ ! -f /etc/apt/sources.list.d/nvidia-container-toolkit.list ]; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    $APT update -y
  fi
  $APT install -y nvidia-container-toolkit
fi
nvidia-ctk runtime configure --runtime=docker
# A systemd reload (snap refreshes trigger them) cuts running containers off their GPUs when Docker uses the
# systemd cgroup driver ("Failed to initialize NVML: Unknown Error"). Use cgroupfs, and stop snapd refreshing.
python3 - <<'PY'
import json, os
p = "/etc/docker/daemon.json"
txt = open(p).read() if os.path.exists(p) else ""
d = json.loads(txt) if txt.strip() else {}
d["exec-opts"] = [o for o in d.get("exec-opts", []) if not o.startswith("native.cgroupdriver=")] + ["native.cgroupdriver=cgroupfs"]
open(p, "w").write(json.dumps(d, indent=2) + "\n")
PY
if snap list 2>/dev/null | grep -q '^docker '; then
  snap refresh --hold || true
else
  systemctl stop snapd.service snapd.socket snapd.seeded.service 2>/dev/null || true
  systemctl mask snapd.service snapd.socket snapd.seeded.service 2>/dev/null || true
fi
systemctl restart docker

say "3/6 Checking the GPUs from inside Docker, including after a systemd reload"
if ! nvidia-smi -L; then
  echo "nvidia-smi failed. If it says 'Driver/library version mismatch', run: reboot"
  echo "then reconnect and run this script again."; exit 1
fi
echo "Docker cgroup driver: $(docker info -f '{{.CgroupDriver}}')"
docker rm -f optimus_gputest >/dev/null 2>&1 || true
docker run -d --name optimus_gputest --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 sleep 300 >/dev/null
systemctl daemon-reload; sleep 5
if ! docker exec optimus_gputest nvidia-smi -L; then
  docker rm -f optimus_gputest >/dev/null 2>&1
  echo "GPU lost after a systemd reload: do not start runs on this machine. Paste this output."; exit 1
fi
docker rm -f optimus_gputest >/dev/null 2>&1

say "4/6 Miniconda"
CONDA="$HOME/miniconda3/bin/conda"
if [ ! -x "$CONDA" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  rm -f /tmp/miniconda.sh
  "$CONDA" init bash >/dev/null
fi
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

say "5/6 NatureBench and its conda environments (10 to 20 minutes the first time)"
cd "$HOME"
[ -d NatureBench ] || git clone https://github.com/FrontisAI/NatureBench.git
cd NatureBench
"$CONDA" env list | awk '{print $1}' | grep -qx naturebench || "$CONDA" env create -f conda_env.yml
"$CONDA" env list | awk '{print $1}' | grep -qx naturebench-eval || "$CONDA" env create -f conda_env_eval.yml

say "6/6 Installing OPTIMUS"
bash "$BUNDLE/install_optimus.sh" "$HOME/NatureBench"
cp "$BUNDLE/run_optimus.sh" "$BUNDLE/status.py" "$BUNDLE/judge_diag.py" "$BUNDLE/alt_judge.py" "$HOME/"
chmod +x "$HOME/run_optimus.sh"

say "Setup complete. Disk:"
df -h / | tail -1
cat <<'EOT'

Next:
  1. Save your OpenRouter key (paste it when asked):
       read -rp "Paste OpenRouter key: " k; printf '%s' "$k" > ~/.openrouter_key; chmod 600 ~/.openrouter_key; unset k; clear
  2. Start tmux:   tmux new -s optimus
  3. Inside tmux:  ~/run_optimus.sh task-set/naturebench-25/gpu_low.txt optimus_nb25_gpu_low 0,1,2,3
EOT
