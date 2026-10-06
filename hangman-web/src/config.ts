export const MAX_TITLE_LENGTH = 80
export const MAX_FAILURES = 8
export const INDEX_LATE_FAILURES = 2
export const MAX_INDEX_CANDIDATES = 200

export const LETTERS = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
] as const

export type Letter = (typeof LETTERS)[number]

const VOCABULARY = ['[PAD]', '[MASK]', ...LETTERS, ' ']

export const CHAR_TO_ID = Object.fromEntries(
  VOCABULARY.map((character, index) => [character, index]),
) as Record<string, number>

export const MASK_ID = CHAR_TO_ID['[MASK]']
export const VOCAB_SIZE = VOCABULARY.length

export const MODEL_URLS = {
  webgpu: `${import.meta.env.BASE_URL}models/slimbert_fp16.onnx`,
  wasm: `${import.meta.env.BASE_URL}models/slimbert.onnx`,
}

export const INDEX_URL = `${import.meta.env.BASE_URL}index.csv`
