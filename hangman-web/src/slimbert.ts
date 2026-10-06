import type * as Ort from 'onnxruntime-web'

import {
  CHAR_TO_ID,
  LETTERS,
  MASK_ID,
  MODEL_URLS,
  VOCAB_SIZE,
  type Letter,
} from './config'
import { loadWordIndex } from './wordIndex'

export type Provider = 'webgpu' | 'wasm'
type PredictionSource = 'model' | 'index'

type PredictionOptions = {
  useIndex: boolean
  indexLateFailures: number
  maxFailures: number
}

export type SlimBertPrediction = {
  letter: Letter
  provider: Provider
  score: number
  source: PredictionSource
}

function encodePattern(pattern: string): number[] {
  return [...pattern].map((character) =>
    character === '_' ? MASK_ID : CHAR_TO_ID[character],
  )
}

function softmaxProbabilities(logits: Float32Array, offset: number): Float32Array {
  let maximum = Number.NEGATIVE_INFINITY

  for (let index = 0; index < VOCAB_SIZE; index += 1) {
    maximum = Math.max(maximum, logits[offset + index])
  }

  let total = 0

  for (let index = 0; index < VOCAB_SIZE; index += 1) {
    total += Math.exp(logits[offset + index] - maximum)
  }

  const probabilities = new Float32Array(VOCAB_SIZE)

  for (let index = 0; index < VOCAB_SIZE; index += 1) {
    probabilities[index] = Math.exp(logits[offset + index] - maximum) / total
  }

  return probabilities
}

export class SlimBertPredictor {
  private runtime?: typeof Ort
  private session?: Ort.InferenceSession
  private activeProvider?: Provider
  private loading?: Promise<void>

  getProvider(): Provider | undefined {
    return this.activeProvider
  }

  async load(): Promise<void> {
    this.loading ??= this.createSession()
    return this.loading
  }

  async predictNextLetter(
    pattern: string,
    missedLetters: ReadonlySet<Letter>,
    guessedLetters: ReadonlySet<Letter>,
    options: PredictionOptions,
  ): Promise<SlimBertPrediction> {
    await this.load()

    const inputIds = encodePattern(pattern)
    const blankPositions = inputIds
      .map((tokenId, index) => (tokenId === MASK_ID ? index : -1))
      .filter((index) => index >= 0)

    if (blankPositions.length === 0) {
      throw new Error('The board has no hidden letters.')
    }

    const outputs = await this.session!.run({
      input_ids: new this.runtime!.Tensor(
        'int64',
        BigInt64Array.from(inputIds, BigInt),
        [1, inputIds.length],
      ),
      missed_letters: new this.runtime!.Tensor(
        'float32',
        Float32Array.from(LETTERS, (letter) => Number(missedLetters.has(letter))),
        [1, LETTERS.length],
      ),
    })
    const mlmLogits = outputs.mlm_logits.data as Float32Array
    const probabilitiesByPosition = new Map<number, Float32Array>(
      blankPositions.map((position) => [
        position,
        softmaxProbabilities(mlmLogits, position * VOCAB_SIZE),
      ]),
    )
    let bestLetter: Letter | undefined
    let bestScore = Number.NEGATIVE_INFINITY
    const modelScores = new Map<Letter, number>()

    for (const letter of LETTERS) {
      if (guessedLetters.has(letter)) {
        continue
      }

      let absentProbability = 1

      for (const position of blankPositions) {
        absentProbability *= 1 - (probabilitiesByPosition.get(position)?.[CHAR_TO_ID[letter]] ?? 0)
      }

      const score = 1 - absentProbability
      modelScores.set(letter, score)

      if (score > bestScore) {
        bestLetter = letter
        bestScore = score
      }
    }

    if (bestLetter === undefined || this.activeProvider === undefined) {
      throw new Error('No unguessed letters remain.')
    }

    let indexLetter: Letter | undefined

    if (options.useIndex && missedLetters.size >= options.maxFailures - options.indexLateFailures) {
      try {
        const wordIndex = await loadWordIndex()
        indexLetter = await wordIndex.guessLetter(
          pattern,
          guessedLetters,
          missedLetters,
          probabilitiesByPosition,
          modelScores,
        )
      } catch {
        indexLetter = undefined
      }
    }

    return {
      letter: indexLetter ?? bestLetter,
      provider: this.activeProvider,
      score: modelScores.get(indexLetter ?? bestLetter) ?? bestScore,
      source: indexLetter ? 'index' : 'model',
    }
  }

  private async createSession(): Promise<void> {
    try {
      const runtime = await import('onnxruntime-web/webgpu')
      const session = await runtime.InferenceSession.create(MODEL_URLS.webgpu, {
        executionProviders: ['webgpu'],
      })

      this.runtime = runtime
      this.session = session
      this.activeProvider = 'webgpu'
    } catch (error) {
      console.warn('WebGPU initialization failed; falling back to FP32/WASM.', error)
      const runtime = await import('onnxruntime-web/wasm')
      const session = await runtime.InferenceSession.create(MODEL_URLS.wasm, {
        executionProviders: ['wasm'],
      })

      this.runtime = runtime
      this.session = session
      this.activeProvider = 'wasm'
    }
  }
}

export const slimbert = new SlimBertPredictor()
