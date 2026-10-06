vcpkg overlay ports for the iOS build.

- `libvpx`: upstream port from the vcpkg baseline in `../vcpkg.json`, patched to build for the arm64 iOS
  simulator (libvpx's own build system only knows the iPhoneOS SDK for arm64 Apple targets).
  Search the portfile for "Ring Racers" to find the change.
