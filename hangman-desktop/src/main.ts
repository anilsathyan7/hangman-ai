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
