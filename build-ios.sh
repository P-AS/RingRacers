#!/bin/bash
# Build and Package Dr. Robotnik's Ring Racers for iOS
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
useassets=true
simulator=false
skipbuild=false
BUNDLE_NAME="RingRacers"
BUNDLE_ID="org.kartkrew.ringracers"
# Keep in sync with VCPKG_OSX_DEPLOYMENT_TARGET in src/sdl/ios/triplets
DEPLOYMENT_TARGET=15.0
ASSET_DIR="assets/installer"
devicename=""

for args in "$@"; do
	case "$args" in
		"--noassets")
			useassets=false
			shift
			;;
		"--skipbuild")
			skipbuild=true
			shift
			;;
		"--simulator"|"-S")
			simulator=true
			shift
			;;
		*)
			devicename="$1"
			shift
			;;
	esac
done

cd "$REPO"

# Game assets are not redistributed with the source code.
# Copy the contents of an existing install's data directory (bios.pk3, data/, models/, ...) into assets/installer.
if [ "$useassets" == true ] && [ ! -f "$ASSET_DIR/bios.pk3" ]; then
	echo "Game assets not found in $ASSET_DIR."
	echo "Copy bios.pk3, data/, models/, models.dat and gamecontrollerdb.txt from a Ring Racers install there,"
	echo "or build with --noassets."
	exit 1
fi

if [ "$simulator" == true ]; then
	echo "Simulator Build"
	TARGET=SIMULATORARM64
	TRIPLET=arm64-ios-simulator
	BUILD_DIR="build-ios/simulator"
else
	echo "Device Build"
	TARGET=OS64
	TRIPLET=arm64-ios
	BUILD_DIR="build-ios/device"
fi

# Dependencies come from vcpkg. Use $VCPKG_ROOT if set, otherwise a checkout under build-ios.
if [ -z "${VCPKG_ROOT:-}" ]; then
	VCPKG_ROOT="$REPO/build-ios/vcpkg"
	if [ ! -d "$VCPKG_ROOT" ]; then
		git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
	fi
fi

if [ "$skipbuild" == true ]; then
	echo "Skipping build"
else
	# configure and build
	cmake -S . -B "$BUILD_DIR" -G Xcode \
		-Wno-deprecated \
		-DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
		-DVCPKG_CHAINLOAD_TOOLCHAIN_FILE="$REPO/cmake/Modules/ios.toolchain.cmake" \
		-DVCPKG_TARGET_TRIPLET="$TRIPLET" \
		-DVCPKG_MANIFEST_DIR="$REPO/src/sdl/ios" \
		-DVCPKG_OVERLAY_TRIPLETS="$REPO/src/sdl/ios/triplets" \
		-DVCPKG_OVERLAY_PORTS="$REPO/src/sdl/ios/overlay-ports" \
		-DPLATFORM="$TARGET" \
		-DDEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
		-DSRB2_CONFIG_EXTERNAL_ASSETS="$useassets" \
		-DSRB2_ASSET_DIRECTORY="$REPO/$ASSET_DIR"

	cmake --build "$BUILD_DIR" --config Release -- -quiet -allowProvisioningUpdates
fi

# Package the app, with assets if enabled
rm -rf "$BUILD_DIR/bin/Release"
cmake --install "$BUILD_DIR" --config Release --prefix "$BUILD_DIR/bin/Release"

# Insert app icon
PARTIAL_PLIST="$(mktemp)"
xcrun actool --compile "$REPO/$BUILD_DIR/bin/Release/$BUNDLE_NAME.app" \
	--platform "$([ "$simulator" == true ] && echo iphonesimulator || echo iphoneos)" \
	--minimum-deployment-target "$DEPLOYMENT_TARGET" \
	--app-icon ringracers \
	--output-partial-info-plist "$PARTIAL_PLIST" \
	"$REPO/src/sdl/ringracers.icon" \
	> /dev/null

/usr/libexec/PlistBuddy -c "Merge $PARTIAL_PLIST :" "$REPO/$BUILD_DIR/bin/Release/$BUNDLE_NAME.app/Info.plist"
rm -f "$PARTIAL_PLIST"

# Installing and adding the icon modified the bundle, so sign it again
if [ "$simulator" == true ]; then
	codesign --force --sign - "$BUILD_DIR/bin/Release/$BUNDLE_NAME.app"
fi

if [ "$simulator" == false ]; then
	# Package IPA for distribution
	mkdir -p "$BUILD_DIR/bin/Release/Payload"
	cp -R "$BUILD_DIR/bin/Release/$BUNDLE_NAME.app" "$BUILD_DIR/bin/Release/Payload"
	cd "$BUILD_DIR/bin/Release"
	COPYFILE_DISABLE=1 zip -qr "./$BUNDLE_NAME.ipa" Payload -x '.*'
	cd "$REPO"
	rm -rf "$BUILD_DIR/bin/Release/Payload"
else
	# Launch in Device Hub
	BOOTED_UDID="$(xcrun simctl list devices booted -j | /usr/bin/python3 -c \
	'import json,sys; d=json.load(sys.stdin)["devices"]; print(next((dev["udid"] for devs in d.values() for dev in devs), ""))')"
	if [ -z "$BOOTED_UDID" ]; then
		xcrun simctl boot "$devicename"
		if [ -d "/Applications/Xcode.app/Contents/Applications/DeviceHub.app" ]; then
			open "/Applications/Xcode.app/Contents/Applications/DeviceHub.app" 2>/dev/null || true
		else
			open -a Simulator 2>/dev/null || true
		fi
		# Give the simulator a moment to finish booting before installing.
		xcrun simctl bootstatus "$devicename" -b
	fi
	xcrun simctl install booted "$BUILD_DIR/bin/Release/$BUNDLE_NAME.app"
	xcrun simctl launch --console booted "$BUNDLE_ID"
fi
