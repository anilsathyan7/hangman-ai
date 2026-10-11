#!/usr/bin/env bash
set -euo pipefail

# Install the selected build; optionally launch it as your normal user.
launch=false
deb=''
usage='Usage: bash install-electron.sh [path/to/app.deb] [--launch]'
for arg in "$@"; do
  case "$arg" in
    --launch) launch=true ;;
    -*|'') printf '%s\n' "$usage" >&2; exit 2 ;;
    *)
      if [[ -n "$deb" ]]; then
        printf '%s\n' "$usage" >&2
        exit 2
      fi
      deb="$arg"
      ;;
  esac
done

if [[ -z "$deb" ]]; then
  script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  version=$(node -p 'require(process.argv[1]).version' -- "$script_dir/../hangman-desktop/package.json")
  deb_arch=$(dpkg --print-architecture)
  deb="$PWD/hangman-desktop_${version}_${deb_arch}.deb"
fi

if [[ ! -f "$deb" ]]; then
  printf 'Package missing: %s\nRun bash build-electron.sh first.\n' "$deb" >&2
  exit 1
fi
deb=$(realpath -- "$deb")
app_name=$(dpkg-deb --field "$deb" Package)
if [[ ! "$app_name" =~ ^[a-z0-9][a-z0-9+.-]+$ ]]; then
  printf 'Invalid DEB package name.\n' >&2
  exit 1
fi

# Reinstall also refreshes an existing app when its version number is unchanged.
sudo apt-get install --reinstall "$deb"
if [[ "$launch" == true ]]; then
  exec "$app_name" --ozone-platform=x11
fi
