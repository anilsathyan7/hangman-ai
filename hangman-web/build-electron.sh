#!/usr/bin/env bash
set -euo pipefail

# Create a sibling Electron wrapper and build from the selected Vite web source.
source_path=''
requested_app_name=''
output_name=''
requested_version=''
cleanup=false
usage='Usage: bash build-electron.sh [path/to/vite-web-app] [--app-name lowercase-name] [--version X.Y.Z] [--output-name app.deb] [--cleanup]'
usage_error() {
  printf '%s\n' "$usage" >&2
  exit 2
}
while (($#)); do
  case "$1" in
    --cleanup) cleanup=true; shift ;;
    --app-name)
      requested_app_name="${2:-}"
      [[ "$requested_app_name" =~ ^[a-z0-9][a-z0-9-]+$ && ${#requested_app_name} -le 214 ]] || usage_error
      shift 2
      ;;
    --output-name)
      output_name="${2:-}"
      [[ "$output_name" == *.deb && "$output_name" != */* && "$output_name" != -* ]] || usage_error
      shift 2
      ;;
    --version)
      requested_version="${2:-}"
      [[ "$requested_version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] || usage_error
      shift 2
      ;;
    -*|'') usage_error ;;
    *)
      [[ -z "$source_path" ]] || usage_error
      source_path="$1"
      shift
      ;;
  esac
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output_dir="$PWD"
source_dir=$(cd -- "${source_path:-$script_dir}" && pwd)
for config in vite.config.ts vite.config.mts vite.config.js vite.config.mjs vite.config.cts vite.config.cjs; do
  [[ -f "$source_dir/$config" ]] && break
done
if [[ ! -f "$source_dir/package.json" || ! -f "$source_dir/package-lock.json" ||
      ! -f "$source_dir/index.html" || ! -f "$source_dir/$config" ]]; then
  printf 'Expected a Vite frontend with package.json, package-lock.json, index.html and vite.config.*: %s\n' "$source_dir" >&2
  exit 1
fi
source_name=$(basename -- "$source_dir")
app_name="$requested_app_name"
if [[ -z "$app_name" ]]; then
  app_name=$(node -p "(process.argv[1].toLowerCase().replace(/-web$/, '').replace(/[^a-z0-9-]+/g, '-').replace(/^-+|-+$/g, '') || 'app') + '-desktop'" -- "$source_name")
fi
desktop_dir="$(dirname -- "$source_dir")/$app_name"
if [[ "$desktop_dir" == "$source_dir" ]]; then
  printf 'Choose an app name different from the web source folder name.\n' >&2
  exit 2
fi
source_ref=$(node -p 'JSON.stringify("../" + process.argv[1])' -- "$source_name")
config_ref=$(node -p 'JSON.stringify("../" + process.argv[1] + "/" + process.argv[2])' -- "$source_name" "$config")
cd "$source_dir"
npm ci

if [[ ! -f "$desktop_dir/package.json" ]]; then
  cd "$(dirname -- "$source_dir")"
  npx --yes create-electron-app@8.0.1 "$app_name" \
    --template vite --typescript --package-manager npm@10.9.9 --electron-version 44.7.0

  cat > "$desktop_dir/vite.renderer.config.mts" <<EOF
import path from 'node:path';
import { defineConfig, mergeConfig, type UserConfigExport } from 'vite';
import webConfig from $config_ref;

export default defineConfig(async (env) => {
  const exported = webConfig as unknown as UserConfigExport;
  const config = typeof exported === 'function' ? await exported(env) : await exported;
  return mergeConfig(config, {
    root: path.resolve(import.meta.dirname, $source_ref),
    base: './',
    build: { outDir: path.resolve(import.meta.dirname, '.vite/renderer/main_window') },
  });
});
EOF

  cat > "$desktop_dir/src/main.ts" <<'EOF'
import { app, BrowserWindow, net, protocol } from 'electron';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import started from 'electron-squirrel-startup';

if (started) app.quit();
protocol.registerSchemesAsPrivileged([{
  scheme: 'app',
  privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true },
}]);

function createWindow() {
  const window = new BrowserWindow({
    title: app.getName(), width: 1100, height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      nodeIntegration: false, contextIsolation: true, sandbox: true,
    },
  });
  window.on('page-title-updated', (event) => event.preventDefault());
  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  void window.loadURL(MAIN_WINDOW_VITE_DEV_SERVER_URL || 'app://bundle/index.html');
}

app.whenReady().then(() => {
  if (started) return;
  const root = path.resolve(__dirname, '../renderer', MAIN_WINDOW_VITE_NAME);
  protocol.handle('app', async (request) => {
    try {
      const url = new URL(request.url);
      const file = path.resolve(root, '.' + decodeURIComponent(url.pathname));
      const relative = path.relative(root, file);
      if (url.host !== 'bundle' || relative === '..' ||
          relative.startsWith('..' + path.sep) || path.isAbsolute(relative)) {
        return new Response('Forbidden', { status: 403 });
      }
      return await net.fetch(pathToFileURL(file).href);
    } catch {
      return new Response('Not found', { status: 404 });
    }
  });
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
EOF
else
  if [[ ! -f "$desktop_dir/vite.renderer.config.mts" ]] ||
      [[ "$(cat "$desktop_dir/vite.renderer.config.mts")" != *"${config_ref:1:${#config_ref}-2}"* ]]; then
    printf 'The existing %s folder must be configured to use %s.\n' "$app_name" "$source_dir" >&2
    exit 1
  fi
  cd "$desktop_dir"
  npm ci
fi

cd "$desktop_dir"
if [[ -n "$requested_version" ]]; then
  npm version "$requested_version" --no-git-tag-version --allow-same-version --ignore-scripts
fi
npm pkg set description="$app_name desktop application" \
  'scripts.start=electron-forge start -- --ozone-platform=x11' \
  'scripts.typecheck=tsc --noEmit --allowImportingTsExtensions'
npm run typecheck

npm run make -- --targets=@electron-forge/maker-deb
version=$(node -p "require('./package.json').version")
arch=$(node -p 'process.arch')
deb_arch=$(dpkg --print-architecture)
deb="$desktop_dir/out/make/deb/$arch/${app_name}_${version}_${deb_arch}.deb"
if [[ ! -s "$deb" ]]; then
  printf 'Expected DEB missing or empty; build files kept: %s\n' "$deb" >&2
  exit 1
fi

# Move the DEB to the directory where the command was run before cleaning up.
final_deb="$output_dir/${output_name:-$(basename -- "$deb")}"
if [[ "$deb" != "$final_deb" ]]; then
  mv -f -- "$deb" "$final_deb"
fi

printf 'DEB ready: %s\n' "$final_deb"
if [[ "$cleanup" == true ]]; then
  # Keep the wrapper source and any directory containing the final DEB.
  for build_dir in "$desktop_dir/.vite" "$desktop_dir/out/$app_name-linux-$arch" "$desktop_dir/out/make-results"; do
    if [[ "$final_deb" == "$build_dir/"* ]]; then
      printf 'Kept directory containing the final DEB: %s\n' "$build_dir"
    else
      rm -rf -- "$build_dir"
    fi
  done
  printf 'Build cleanup complete.\n'
else
  printf 'Intermediate build files kept. Use --cleanup to remove them after a build.\n'
fi
