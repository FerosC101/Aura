#!/usr/bin/env bash
# Capture presentation mockups of AURA at native iPhone resolution.
#
#   tools/mockups.sh [device name]
#
# Writes PNGs to ios/screenshots/presentation/. These are device-sized frames
# straight from the simulator -- no window chrome, no scaling -- so they can be
# dropped into slides or a device frame without resampling.
#
# Distinct from tools/screenshots.sh, which refreshes the smaller set embedded
# in the README.
set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
NAME="${1:-iPhone 17 Pro}"
DEVICE=$(xcrun simctl list devices available | grep -E "^    $NAME \(" | tail -1 | grep -oE "[0-9A-F-]{36}")
OUT=screenshots/presentation
BUNDLE=com.aura.selectiveawareness
VOICES="Dr. Reyes,Maria,Ben"
mkdir -p "$OUT"

xcodebuild -project Aura.xcodeproj -scheme Aura -configuration Debug \
  -destination "platform=iOS Simulator,id=$DEVICE" -derivedDataPath build/dd \
  CODE_SIGNING_ALLOWED=NO build -quiet
APP=$(find build/dd/Build/Products/Debug-iphonesimulator -maxdepth 1 -name 'Aura.app' | head -1)

xcrun simctl boot "$DEVICE" 2>/dev/null || true
xcrun simctl bootstatus "$DEVICE" -b >/dev/null
# A fixed status bar keeps every frame identical across captures.
xcrun simctl status_bar "$DEVICE" override \
  --time "9:41" --batteryState charged --batteryLevel 100 --cellularBars 4 --wifiBars 3
xcrun simctl install "$DEVICE" "$APP"

shot() {  # name, launch args...
  local name=$1; shift
  xcrun simctl terminate "$DEVICE" "$BUNDLE" 2>/dev/null || true
  # A fresh container each time, so one capture's state cannot leak into the next.
  xcrun simctl launch "$DEVICE" "$BUNDLE" "$@" >/dev/null
  sleep "${WAIT:-3}"
  xcrun simctl io "$DEVICE" screenshot --type=png "$OUT/$name.png" >/dev/null 2>&1
  printf '  %-28s %s\n' "$name.png" "$(sips -g pixelWidth -g pixelHeight "$OUT/$name.png" \
    | awk '/pixel/{printf "%s ", $2}')"
}

echo "Capturing to $OUT (device $NAME)"

# -- the story, in the order a deck would tell it ---------------------------
for step in 0 1 2 3 4 5; do shot "01-onboarding-$step" -AuraOnboarding "$step"; done

shot            "02-home-paused"   -AuraName Vince -AuraMode care -AuraScreen home
WAIT=7 shot     "03-home-listening" -AuraName Vince -AuraMode care -AuraScreen home -AuraDemo clinic
WAIT=6 shot     "04-voice-focus"   -AuraName Vince -AuraMode care -AuraScreen home \
                                   -AuraSheet voicefocus -AuraVoices "$VOICES" -AuraDemo clinic
shot            "05-modes"         -AuraName Vince -AuraMode care -AuraScreen modes
WAIT=9 shot     "06-insights"      -AuraName Vince -AuraMode care -AuraScreen insights -AuraDemo clinic
WAIT=8 shot     "07-demo-clinic"   -AuraName Vince -AuraScreen demo -AuraDemo clinic
WAIT=8 shot     "08-demo-commute"  -AuraName Vince -AuraScreen demo -AuraDemo psc_demo
shot            "09-you"           -AuraName Vince -AuraMode care -AuraScreen you -AuraVoices "$VOICES"

xcrun simctl terminate "$DEVICE" "$BUNDLE" 2>/dev/null || true
echo
echo "$(ls -1 "$OUT"/*.png | wc -l | tr -d ' ') frames in $OUT"
