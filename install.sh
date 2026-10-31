#!/bin/zsh
# Builds the helper, creates the venv, and installs a LaunchAgent that runs at login and restarts on crash.
set -e
D=${0:A:h}; cd "$D"
PY=${PYTHON:-$(command -v python3)}
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' || { echo "need Python 3.10+ (set PYTHON=...)"; exit 1; }
command -v swiftc >/dev/null || { echo "need swiftc: xcode-select --install"; exit 1; }

swiftc -O airpods_ca.swift -o airpods_ca -framework IOBluetooth
swiftc -O sysaudio.swift -o sysaudio -framework ScreenCaptureKit -framework CoreMedia
mkdir -p models
[[ -f models/wespeaker_en_voxceleb_resnet34.onnx ]] || curl -fsSL -o models/wespeaker_en_voxceleb_resnet34.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx
"$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

PLIST=~/Library/LaunchAgents/co.noxmetals.encore.plist
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>co.noxmetals.encore</string>
  <key>ProgramArguments</key><array>
    <string>$D/.venv/bin/python</string><string>$D/encore.py</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$D/encore.log</string>
  <key>StandardErrorPath</key><string>$D/encore.log</string>
</dict></plist>
EOF
launchctl bootout gui/$(id -u) "$PLIST" 2>/dev/null || true
launchctl bootstrap gui/$(id -u) "$PLIST"
echo "installed; log: $D/encore.log"
.venv/bin/python encore.py --doctor || true
