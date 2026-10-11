import path from 'node:path';
import { defineConfig, mergeConfig, type UserConfigExport } from 'vite';
import webConfig from "../hangman-web/vite.config.ts";

export default defineConfig(async (env) => {
  const exported = webConfig as unknown as UserConfigExport;
  const config = typeof exported === 'function' ? await exported(env) : await exported;
  return mergeConfig(config, {
    root: path.resolve(import.meta.dirname, "../hangman-web"),
    base: './',
    build: { outDir: path.resolve(import.meta.dirname, '.vite/renderer/main_window') },
  });
});
