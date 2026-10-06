import { ChevronDown, RotateCcw, SlidersHorizontal, ThumbsDown, ThumbsUp } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  INDEX_LATE_FAILURES,
  MAX_FAILURES,
  MAX_TITLE_LENGTH,
  type Letter,
} from './config'
import { createGameState, recordMiss, saveBoard, type GameState } from './game'
import { slimbert, type Provider } from './slimbert'
import './App.css'

type GamePhase = 'setup' | 'predicting' | 'feedback' | 'placing' | 'ready' | 'finished'

function normalizePattern(value: string) {
  return value.toUpperCase().replace(/[^A-Z_ ]/g, '').slice(0, MAX_TITLE_LENGTH)
}

function App() {
  const [pattern, setPattern] = useState('')
  const [initialPattern, setInitialPattern] = useState<string | null>(null)
  const [gameState, setGameState] = useState<GameState | null>(null)
  const [pendingPattern, setPendingPattern] = useState<string | null>(null)
  const [suggestedLetter, setSuggestedLetter] = useState<Letter | null>(null)
  const [phase, setPhase] = useState<GamePhase>('setup')
  const [modelStatus, setModelStatus] = useState('Loading AI...')
  const [activeProvider, setActiveProvider] = useState<Provider | 'loading' | 'unavailable'>('loading')
  const [useIndex, setUseIndex] = useState(false)
  const [maxFailures, setMaxFailures] = useState(MAX_FAILURES)
  const [indexLateFailures, setIndexLateFailures] = useState(INDEX_LATE_FAILURES)
  const [showSettings, setShowSettings] = useState(false)
  const [showGuide, setShowGuide] = useState(false)
  const predictionRequest = useRef(0)
  const words = useMemo(() => pattern.trim().split(/\s+/).filter(Boolean), [pattern])
  const hasVisibleLetterInEveryWord = words.every((word) => /[A-Z]/.test(word))
  const isReady = words.length > 0 && pattern.includes('_') && hasVisibleLetterInEveryWord
  const boardPattern = pendingPattern ?? gameState?.pattern ?? pattern
  const isSetup = phase === 'setup'
  const canSaveBoard = pendingPattern !== null && pendingPattern !== gameState?.pattern
  const resultClass = phase === 'finished'
    ? modelStatus === 'Solved!'
      ? ' success'
      : ' failure'
    : ''
  const correctLetters = gameState
    ? [...gameState.guessedLetters].filter((letter) => !gameState.missedLetters.has(letter))
    : []

  useEffect(() => {
    slimbert.load()
      .then(() => {
        setActiveProvider(slimbert.getProvider() ?? 'unavailable')
        setModelStatus('AI Ready')
      })
      .catch(() => {
        setActiveProvider('unavailable')
        setModelStatus('AI could not load')
      })
  }, [])

  function addToPattern(value: string) {
    setPattern((currentPattern) => normalizePattern(currentPattern + value))
  }

  function handleBoardKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    if (!isSetup) {
      return
    }

    if (event.key === 'Backspace') {
      event.preventDefault()
      setPattern((currentPattern) => currentPattern.slice(0, -1))
      return
    }

    if (/^[A-Z_ ]$/i.test(event.key)) {
      event.preventDefault()
      addToPattern(event.key)
    }
  }

  function handleBoardPaste(event: React.ClipboardEvent<HTMLDivElement>) {
    if (!isSetup) {
      return
    }

    event.preventDefault()
    addToPattern(event.clipboardData.getData('text'))
  }

  async function requestNextGuess(state: GameState) {
    const requestId = ++predictionRequest.current

    if (!state.pattern.includes('_')) {
      setSuggestedLetter(null)
      setModelStatus('Solved!')
      setPhase('finished')
      return
    }

    if (state.missedLetters.size >= maxFailures) {
      setSuggestedLetter(null)
      setModelStatus(`Game Over: ${maxFailures} missed letters.`)
      setPhase('finished')
      return
    }

    setSuggestedLetter(null)
    setPhase('predicting')
    setModelStatus('AI choosing the next letter...')

    try {
      const prediction = await slimbert.predictNextLetter(
        state.pattern,
        state.missedLetters,
        state.guessedLetters,
        { useIndex, indexLateFailures, maxFailures },
      )
      if (requestId !== predictionRequest.current) {
        return
      }

      setSuggestedLetter(prediction.letter)
      setModelStatus(
        `${prediction.source === 'index' ? 'AI + lookup' : 'AI'} running on ${prediction.provider.toUpperCase()}`,
      )
      setPhase('feedback')
    } catch {
      if (requestId !== predictionRequest.current) {
        return
      }

      setModelStatus('AI could not choose a letter')
      setPhase('ready')
    }
  }

  async function startGame() {
    const initialState = createGameState(pattern)
    setInitialPattern(pattern)
    setGameState(initialState)
    await requestNextGuess(initialState)
  }

  async function restartGame() {
    if (initialPattern === null) {
      return
    }

    const initialState = createGameState(initialPattern)
    setPattern(initialPattern)
    setGameState(initialState)
    setPendingPattern(null)
    await requestNextGuess(initialState)
  }

  function resetGame() {
    predictionRequest.current += 1
    setPattern('')
    setInitialPattern(null)
    setGameState(null)
    setPendingPattern(null)
    setSuggestedLetter(null)
    setPhase('setup')
    setModelStatus('Set a new board to start.')
  }

  function markCorrect() {
    setPendingPattern(gameState!.pattern)
    setPhase('placing')
    setModelStatus(`Place every ${suggestedLetter} on the board, then save it.`)
  }

  async function markWrong() {
    const nextState = recordMiss(gameState!, suggestedLetter!)
    setGameState(nextState)
    await requestNextGuess(nextState)
  }

  function updateTile(index: number) {
    if (!gameState || !pendingPattern || !suggestedLetter) {
      return
    }

    const currentCharacter = pendingPattern[index]

    if (currentCharacter !== '_' && currentCharacter !== suggestedLetter) {
      return
    }

    const nextPattern = `${pendingPattern.slice(0, index)}${
      currentCharacter === '_' ? suggestedLetter : '_'
    }${pendingPattern.slice(index + 1)}`

    if (!nextPattern.includes('_')) {
      setGameState(saveBoard(gameState, nextPattern))
      setPendingPattern(null)
      setSuggestedLetter(null)
      setModelStatus('Solved!')
      setPhase('finished')
      return
    }

    setPendingPattern(nextPattern)
  }

  async function confirmBoard() {
    if (!gameState || !pendingPattern) {
      return
    }

    const nextState = saveBoard(gameState, pendingPattern)
    setGameState(nextState)
    setPendingPattern(null)
    setSuggestedLetter(null)
    await requestNextGuess(nextState)
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <h1>Hangman AI</h1>
        <p className="tagline">You know the movie! Can the AI figure it out?</p>
        <p className="app-intro">
          Hangman is a letter-guessing game: uncover a hidden word or phrase before
          running out of misses. Here, you pick a movie title, reveal a few letters,
          and challenge the AI to solve it. You hold the clues. The AI plays detective.
        </p>
      </header>

      <div className="guide-panel">
        <button
          aria-controls="game-guide"
          aria-expanded={showGuide}
          className="settings-button guide-toggle"
          onClick={() => setShowGuide((isVisible) => !isVisible)}
          type="button"
        >
          GAME
          <ChevronDown aria-hidden="true" size={16} />
        </button>
        <div className="game-guide" hidden={!showGuide} id="game-guide">
        <section className="setup-copy" aria-labelledby="setup-title">
          <h2 id="setup-title">Game Play</h2>
          <ol className="play-steps">
            <li><strong>Set the board</strong> - Type letters, <code>_</code> for blanks, spaces between words. Backspace to delete.</li>
            <li><strong>Check the guess</strong> - Mark the AI’s suggested letter correct or wrong.</li>
            <li><strong>Place correct letters</strong> - Fill every matching blank, then save to get the next guess.</li>
            <li><strong>Reset and restart</strong> - Reset clears the board. Restart retries the original board from scratch.</li>
          </ol>
        </section>
        <section className="play-rules" aria-labelledby="rules-title">
          <h2 id="rules-title">Game Rules</h2>
          <ul>
            <li>Use <strong>A–Z</strong> letters, underscores for blanks, and spaces only.</li>
            <li>Maximum <strong>{MAX_TITLE_LENGTH}</strong> characters, including blanks and spaces.</li>
            <li>Start with at least <strong>one</strong> revealed letter in <strong>every</strong> word.</li>
            <li>Reveal <strong>every</strong> occurrence of a correct letter.</li>
            <li>The AI must solve the title before <strong>{maxFailures}</strong> wrong guesses.</li>
            <li>Use <strong>english</strong> movies for best results.</li>
          </ul>
        </section>
        </div>
      </div>

      <section className="game-setup" aria-label="Hangman game">
        <div className="board-area">
          <div
            className={`board-panel${isSetup ? ' board-editor' : ''}`}
            aria-label={isSetup ? 'Title pattern editor' : 'Current Hangman board'}
            aria-live="polite"
            onKeyDown={isSetup ? handleBoardKeyDown : undefined}
            onPaste={isSetup ? handleBoardPaste : undefined}
            role={isSetup ? 'textbox' : undefined}
            tabIndex={isSetup ? 0 : undefined}
          >
            <p className="panel-label">{isSetup ? 'Board preview' : 'Hangman board'}</p>
            <div className="board" aria-label={boardPattern || 'Empty Hangman board'}>
              {boardPattern ? (
                [...boardPattern].map((character, index) =>
                  character === ' ' ? (
                    <span className="word-gap" aria-hidden="true" key={`${character}-${index}`} />
                  ) : phase === 'placing' && (
                    character === '_' || character === suggestedLetter
                  ) ? (
                    <button
                      aria-label={
                        character === '_'
                          ? `Place ${suggestedLetter} in position ${index + 1}`
                          : `Remove ${suggestedLetter} from position ${index + 1}`
                      }
                      className={`letter-tile editable${character === '_' ? ' blank' : ''}`}
                      key={`${character}-${index}`}
                      onClick={() => updateTile(index)}
                      title={character === '_' ? `Place ${suggestedLetter}` : `Remove ${suggestedLetter}`}
                      type="button"
                    >
                      {character === '_' ? '' : character}
                    </button>
                  ) : (
                    <span
                      className={`letter-tile${character === '_' ? ' blank' : ''}${character !== '_' && (initialPattern ?? pattern)[index] === character ? ' initial-letter' : ''}`}
                      key={`${character}-${index}`}
                    >
                      {character === '_' ? '' : character}
                    </span>
                  ),
                )
              ) : (
              <p className="empty-board">Click here and start typing.</p>
            )}
          </div>
          </div>
          <div className="field-meta">
            <span>A-Z, underscores, and spaces only</span>
            <span>{pattern.length} / {MAX_TITLE_LENGTH}</span>
          </div>
          {isSetup && !isReady && pattern && (
            <p className="validation-message">
              Reveal at least one letter in every word to start.
            </p>
          )}
        </div>

        <div className="turn-controls">
          {phase === 'setup' && (
            <div className="setup-actions">
              <button type="button" disabled={!isReady} onClick={startGame}>
                Start Game
              </button>
              {pattern && (
                <button aria-label="Reset board" className="reset-button" onClick={resetGame} title="Reset board" type="button">
                  <RotateCcw aria-hidden="true" size={16} />
                  Reset
                </button>
              )}
            </div>
          )}
          {phase === 'feedback' && suggestedLetter && (
            <>
              <p className="model-suggestion">AI GUESS: {suggestedLetter}</p>
              <div className="feedback-actions">
                <button aria-label="Correct guess" className="feedback-button correct" onClick={markCorrect} title="Correct guess" type="button">
                  <ThumbsUp aria-hidden="true" size={20} />
                </button>
                <button aria-label="Wrong guess" className="feedback-button wrong" onClick={markWrong} title="Wrong guess" type="button">
                  <ThumbsDown aria-hidden="true" size={20} />
                </button>
              </div>
            </>
          )}
          {phase === 'placing' && (
            <>
              <p className="placement-message">Click every blank that contains {suggestedLetter}.</p>
              <button disabled={!canSaveBoard} onClick={confirmBoard} type="button">
                GUESS NEXT
              </button>
            </>
          )}
          {phase === 'ready' && gameState && (
            <button onClick={() => requestNextGuess(gameState)} type="button">
              GUESS NEXT
            </button>
          )}
          {!isSetup && (
            <div className="setup-actions">
              <button aria-label="Restart from the original board" className="reset-button" onClick={restartGame} title="Restart from the original board" type="button">
                <RotateCcw aria-hidden="true" size={16} />
                Restart
              </button>
              <button aria-label="Reset board" className="reset-button" onClick={resetGame} title="Reset board" type="button">
                <RotateCcw aria-hidden="true" size={16} />
                Reset
              </button>
            </div>
          )}
        </div>
        {gameState && (
          <div className="turn-summary" aria-label="Game state">
            <p><span>Correct</span>{correctLetters.join(', ') || 'None'}</p>
            <p><span>Missed</span>{[...gameState.missedLetters].join(', ') || 'None'}</p>
            <p><span>Failures</span>{gameState.missedLetters.size} / {maxFailures}</p>
          </div>
        )}
        <p className={`model-status${resultClass}`} aria-live="polite">{modelStatus}</p>
      </section>
      <div className="settings-panel">
        {showSettings && (
          <section className="settings" aria-label="Game settings">
            <p className="setting-runtime">
              <span>AI runtime</span>
              <strong aria-live="polite">
                {activeProvider === 'webgpu' ? 'WebGPU (GPU) · FP16'
                  : activeProvider === 'wasm' ? 'WASM (CPU) · FP32'
                  : activeProvider === 'loading' ? 'Loading...' : 'Unavailable'}
              </strong>
            </p>
            <label className="setting-toggle">
              <input
                checked={useIndex}
                onChange={(event) => setUseIndex(event.target.checked)}
                type="checkbox"
              />
              Late-Game Word Lookup
            </label>
            <label className="setting-select">
              Lookup Value
              <select
                onChange={(event) => setIndexLateFailures(Number(event.target.value))}
                value={indexLateFailures}
              >
                {[1, 2, 3, 4].map((remainingFails) => (
                  <option key={remainingFails} value={remainingFails}>{remainingFails}</option>
                ))}
              </select>
            </label>
            <label className="setting-select">
              Max Failures
              <select
                onChange={(event) => setMaxFailures(Number(event.target.value))}
                value={maxFailures}
              >
                {Array.from({ length: 15 }, (_, index) => index + 6).map((limit) => (
                  <option key={limit} value={limit}>{limit}</option>
                ))}
              </select>
            </label>
          </section>
        )}
          <button
            aria-expanded={showSettings}
            className="settings-button"
            onClick={() => setShowSettings((isVisible) => !isVisible)}
            type="button"
          >
            <SlidersHorizontal aria-hidden="true" size={16} />
            Settings
          </button>
      </div>
    </main>
  )
}

export default App
