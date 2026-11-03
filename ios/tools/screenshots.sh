#!/usr/bin/env bash
# Build AURA, install it in a simulator, and capture every screen.
#   tools/screenshots.sh [device name]
set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
NAME="${1:-iPhone 17 Pro}"
# Newest runtime wins when several simulators share a name.
DEVICE=$(xcrun simctl list devices available | grep -E "^    $NAME \(" | tail -1 | grep -oE "[0-9A-F-]{36}")
OUT=build/shots
BUNDLE=com.aura.selectiveawareness
mkdir -p "$OUT"

xcodebuild -project Aura.xcodeproj -scheme Aura -configuration Debug \
  -destination "platform=iOS Simulator,id=$DEVICE" -derivedDataPath build/dd \
  CODE_SIGNING_ALLOWED=NO build -quiet
APP=$(find build/dd/Build/Products/Debug-iphonesimulator -maxdepth 1 -name 'Aura.app' | head -1)

xcrun simctl boot "$DEVICE" 2>/dev/null || true
xcrun simctl bootstatus "$DEVICE" -b >/dev/null
xcrun simctl status_bar "$DEVICE" override --time "9:41" --batteryState charged --batteryLevel 100 --cellularBars 4 --wifiBars 3
xcrun simctl install "$DEVICE" "$APP"

shot() {  # name, launch args...
  local name=$1; shift
  xcrun simctl terminate "$DEVICE" "$BUNDLE" 2>/dev/null || true
  xcrun simctl launch "$DEVICE" "$BUNDLE" "$@" >/dev/null
  sleep "${WAIT:-3}"
  xcrun simctl io "$DEVICE" screenshot --type=png "$OUT/$name.png" >/dev/null
  echo "  $OUT/$name.png"
}

for step in 0 1 2 3 4 5; do shot "onboarding-$step" -AuraOnboarding "$step"; done
shot home        -AuraName Vince -AuraMode focus -AuraScreen home
WAIT=6 shot home-live -AuraName Vince -AuraScreen home -AuraDemo psc_demo
shot modes       -AuraName Vince -AuraScreen modes
WAIT=9 shot insights -AuraName Vince -AuraScreen insights -AuraDemo psc_demo
WAIT=7 shot demo     -AuraName Vince -AuraScreen demo -AuraDemo psc_demo
shot you         -AuraName Vince -AuraScreen you
xcrun simctl terminate "$DEVICE" "$BUNDLE" 2>/dev/null || true
