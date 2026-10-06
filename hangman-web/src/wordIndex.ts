import {
  CHAR_TO_ID,
  INDEX_URL,
  MAX_INDEX_CANDIDATES,
  type Letter,
} from './config'

type WordPattern = {
  start: number
  value: string
}

function wordPatterns(pattern: string): WordPattern[] {
  return [...pattern.matchAll(/[^ ]+/g)].map((match) => ({
    start: match.index ?? 0,
    value: match[0],
  }))
}

export class HangmanWordIndex {
  private readonly wordsByLength = new Map<number, string[]>()

  constructor(words: string[]) {
    for (const word of words) {
      const wordsOfLength = this.wordsByLength.get(word.length) ?? []
      wordsOfLength.push(word)
      this.wordsByLength.set(word.length, wordsOfLength)
    }
  }

  async guessLetter(
    pattern: string,
    guessed: ReadonlySet<Letter>,
    missed: ReadonlySet<Letter>,
    probabilitiesByPosition: ReadonlyMap<number, Float32Array>,
    modelScores: ReadonlyMap<Letter, number>,
  ): Promise<Letter | undefined> {
    let bestLetters: Set<Letter> | undefined
    let bestScore = Number.NEGATIVE_INFINITY

    for (const wordPattern of wordPatterns(pattern)) {
      if (!wordPattern.value.includes('_')) {
        continue
      }

      const candidates = this.candidates(wordPattern.value, missed)

      if (candidates.length === 0 || candidates.length > MAX_INDEX_CANDIDATES) {
        continue
      }

      for (const candidate of candidates) {
        let score = 0
        let blanks = 0
        const candidateLetters = new Set<Letter>()

        for (let index = 0; index < wordPattern.value.length; index += 1) {
          if (wordPattern.value[index] !== '_') {
            continue
          }

          const letter = candidate[index] as Letter

          if (guessed.has(letter)) {
            candidateLetters.clear()
            break
          }

          const probabilities = probabilitiesByPosition.get(wordPattern.start + index)

          if (!probabilities) {
            candidateLetters.clear()
            break
          }

          score += Math.log(Math.max(probabilities[CHAR_TO_ID[letter]], 1e-12))
          blanks += 1
          candidateLetters.add(letter)
        }

        if (candidateLetters.size === 0) {
          continue
        }

        score /= blanks

        if (score > bestScore) {
          bestScore = score
          bestLetters = candidateLetters
        }
      }
    }

    if (!bestLetters) {
      return undefined
    }

    return [...bestLetters].reduce((bestLetter, letter) =>
      (modelScores.get(letter) ?? 0) > (modelScores.get(bestLetter) ?? 0)
        ? letter
        : bestLetter,
    )
  }

  private candidates(pattern: string, missed: ReadonlySet<Letter>): string[] {
    return (this.wordsByLength.get(pattern.length) ?? []).filter((word) => {
      for (let index = 0; index < pattern.length; index += 1) {
        if (pattern[index] !== '_' && pattern[index] !== word[index]) {
          return false
        }
      }

      return [...missed].every((letter) => !word.includes(letter))
    })
  }
}

let indexPromise: Promise<HangmanWordIndex> | undefined

export function loadWordIndex(): Promise<HangmanWordIndex> {
  indexPromise ??= fetch(INDEX_URL)
    .then((response) => response.text())
    .then((csv) => new HangmanWordIndex(csv.trim().split(/\r?\n/).slice(1)))

  return indexPromise
}
