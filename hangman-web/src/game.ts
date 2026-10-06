import type { Letter } from './config'

export type GameState = {
  pattern: string
  guessedLetters: Set<Letter>
  missedLetters: Set<Letter>
}

export function createGameState(pattern: string): GameState {
  const guessedLetters = new Set(
    [...pattern].filter((character): character is Letter => /[A-Z]/.test(character)),
  )

  return {
    pattern,
    guessedLetters,
    missedLetters: new Set(),
  }
}

export function recordMiss(state: GameState, letter: Letter): GameState {
  return {
    ...state,
    guessedLetters: new Set([...state.guessedLetters, letter]),
    missedLetters: new Set([...state.missedLetters, letter]),
  }
}

export function saveBoard(state: GameState, pattern: string): GameState {
  const revealedLetters = [...pattern].filter(
    (character): character is Letter => /[A-Z]/.test(character),
  )

  return {
    ...state,
    pattern,
    guessedLetters: new Set([...state.guessedLetters, ...revealedLetters]),
  }
}
