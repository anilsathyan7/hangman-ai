# Hangman AI

You know the movie! Can the AI figure it out? Pick a movie title, reveal a few letters, and challenge the AI to solve it before running out of misses. The AI runs in your browser.

Play online: [Hangman AI](https://hangman-rho-ochre.vercel.app/)

## Install and run

Requires Node.js 20.19+ (20.x) or 22.12+ and npm. From the project folder:

```sh
npm install
npm run dev
```

Open the local URL printed in the terminal. For a production build, run `npm run build`; preview it with `npm run preview`.

## Deploy to Vercel

From the project folder:

```sh
npx vercel login
npx vercel --prod
```

Follow the setup prompts; choose **N** for Git integration to deploy manually. Keep the hidden `.vercel/` folder and run `npx vercel --prod` again after changes to update the same site.

## Desktop application

Build an Electron desktop app for Ubuntu/Debian. Run from the parent `hangman/` folder:

```sh
# Build
bash hangman-web/build-electron.sh hangman-web \
  --app-name hangman-desktop \
  --version 1.0.0 \
  --cleanup

# Install and launch
bash hangman-web/install-electron.sh ./hangman-desktop_1.0.0_amd64.deb --launch
```

| Argument or option | Purpose |
| --- | --- |
| Web app directory | Source folder to build. |
| `--app-name` | Desktop package and wrapper name. |
| `--version X.Y.Z` | Desktop package version. |
| `--output-name app.deb` | Custom DEB filename. |
| `--cleanup` | Remove intermediate files after a successful build. |
| DEB path | Package to install. |
| `--launch` | Launch after installation using X11. |

## Game Play

1. **Set the board** - Type letters, `_` for blanks, spaces between words. Backspace to delete.
2. **Check the guess** - Mark the AI’s suggested letter correct or wrong.
3. **Place correct letters** - Fill every matching blank, then save to get the next guess.
4. **Reset and restart** - Reset clears the board. Restart retries the original board from scratch.

## Game Rules

- Use **A–Z** letters, underscores for blanks, and spaces only.
- Maximum **80** characters, including blanks and spaces.
- Start with at least **one** revealed letter in **every** word.
- Reveal **every** occurrence of a correct letter.
- The AI must solve the title before **8** wrong guesses by default. You can change this limit in Settings.
- Use **english** movies for best results.
