#!/bin/zsh
# Builds the helper, creates the venv, and installs a LaunchAgent that runs at login and restarts on crash.
set -e
D=${0:A:h}; cd "$D"
PY=""  # first Python 3.10+ found; PYTHON=... overrides
for c in $PYTHON python3.14 python3.13 python3.12 python3.11 python3.10 /opt/homebrew/bin/python3 python3; do
  command -v $c >/dev/null && $c -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null && { PY=$c; break; }
done
[[ -n $PY ]] || { echo "need Python 3.10+ (brew install python, or set PYTHON=...)"; exit 1; }
command -v swiftc >/dev/null || { echo "need swiftc: xcode-select --install"; exit 1; }

swiftc -O airpods_ca.swift -o airpods_ca -framework IOBluetooth
swiftc -O sysaudio.swift -o sysaudio -framework ScreenCaptureKit -framework CoreMedia
mkdir -p models
[[ -f models/wespeaker_en_voxceleb_resnet34.onnx ]] || curl -fsSL -o models/wespeaker_en_voxceleb_resnet34.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx
"$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

PLIST=~/Library/LaunchAgents/co.noxmetals.shower-mode.plist
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>co.noxmetals.shower-mode</string>
  <key>ProgramArguments</key><array>
    <string>$D/.venv/bin/python</string><string>$D/shower_mode.py</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$D/shower-mode.log</string>
  <key>StandardErrorPath</key><string>$D/shower-mode.log</string>
</dict></plist>
EOF
launchctl bootout gui/$(id -u) "$PLIST" 2>/dev/null || true
launchctl bootstrap gui/$(id -u) "$PLIST"
echo "installed; log: $D/shower-mode.log"
.venv/bin/python shower_mode.py --doctor || true
