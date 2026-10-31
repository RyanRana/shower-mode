#!/bin/zsh
# Stops the service, removes the LaunchAgent, and turns Conversation Awareness back on.
D=${0:A:h}
PLIST=~/Library/LaunchAgents/co.noxmetals.shower-mode.plist
launchctl bootout gui/$(id -u) "$PLIST" 2>/dev/null || true
rm -f "$PLIST"
[[ -x "$D/airpods_ca" ]] && "$D/airpods_ca" on 2>/dev/null
echo "uninstalled (files in $D are untouched; delete the folder to remove everything)"
