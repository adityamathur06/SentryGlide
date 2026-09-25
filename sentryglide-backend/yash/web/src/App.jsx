import { useEffect, useRef, useState } from 'react'

const BIN_COPY = {
  RED: { label: 'Red bin', color: '#ec5b52', detail: 'Contaminated recyclable plastics' },
  YELLOW: { label: 'Yellow bin', color: '#e8b840', detail: 'Soiled or anatomical waste' },
  BLUE: { label: 'Blue bin', color: '#4c8ddb', detail: 'Glassware and metallic implants' },
  WHITE: { label: 'White bin', color: '#dbe4df', detail: 'Sharps and needles' },
  UNKNOWN: { label: 'Hold item', color: '#b8a4ff', detail: 'Manual review required' },
}

function UploadIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" /></svg>
}

function App() {
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview])

  const choose = (next) => {
    if (!next) return
    if (!next.type.startsWith('image/')) {
      setError('Choose a JPG, PNG, or WebP image.')
      return
    }
    if (preview) URL.revokeObjectURL(preview)
    setFile(next)
    setPreview(URL.createObjectURL(next))
    setResult(null)
    setError('')
  }

  const analyze = async () => {
    if (!file) return
    setBusy(true)
    setError('')
    const data = new FormData()
    data.append('file', file)
    try {
      const response = await fetch('/api/predict', { method: 'POST', body: data })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail || 'Prediction failed')
      setResult(body)
    } catch (err) {
      setError(err.message === 'Failed to fetch'
        ? 'The prediction API is not running. Start the backend on port 8000.'
        : err.message)
    } finally {
      setBusy(false)
    }
  }

  const reset = () => {
    if (preview) URL.revokeObjectURL(preview)
    setFile(null); setPreview(null); setResult(null); setError('')
  }

  const route = result ? BIN_COPY[result.route] : null

  return <main>
    <header className="topbar">
      <a className="brand" href="#top" aria-label="SentryGlide home">
        <span className="brand-mark"><i /><i /><i /></span>
        <span>SentryGlide</span>
      </a>
      <div className="status"><span /> Local model online</div>
    </header>

    <section className="hero" id="top">
      <div className="eyebrow">Biomedical waste intelligence</div>
      <h1>See the item.<br /><em>Know the route.</em></h1>
      <p>Upload a clear, top-down photo of one waste item. The local vision model will classify it and recommend the safest bin.</p>
      <div className="prototype-note"><strong>Prototype mode</strong> Real-image research dataset · For demonstration only</div>
    </section>

    <section className={`workspace ${result ? 'has-result' : ''}`}>
      <div className="upload-panel">
        <div className="section-label"><span>01</span> Inspection image</div>
        <div
          className={`dropzone ${dragging ? 'dragging' : ''} ${preview ? 'with-image' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); choose(e.dataTransfer.files[0]) }}
          onClick={() => !preview && inputRef.current?.click()}
        >
          <input ref={inputRef} type="file" accept="image/png,image/jpeg,image/webp" hidden onChange={(e) => choose(e.target.files[0])} />
          {preview ? <>
            <img src={preview} alt="Selected waste item" />
            <button className="change" onClick={(e) => { e.stopPropagation(); inputRef.current?.click() }}>Change image</button>
          </> : <div className="drop-copy">
            <span className="upload-icon"><UploadIcon /></span>
            <h2>Drop an image here</h2>
            <p>or click to browse your laptop</p>
            <small>JPG, PNG or WebP · maximum 10 MB</small>
          </div>}
        </div>
        <div className="tips">
          <span>For a better result</span>
          <ul><li>Use one item</li><li>Plain background</li><li>Camera directly above</li></ul>
        </div>
        {error && <div className="error">{error}</div>}
        <button className="analyze" disabled={!file || busy} onClick={analyze}>
          {busy ? <><span className="spinner" /> Analyzing locally…</> : 'Analyze item'}
        </button>
      </div>

      <div className="result-panel">
        <div className="section-label"><span>02</span> Classification result</div>
        {!result ? <div className="empty-result">
          <div className="radar"><span /><span /><span /></div>
          <h2>Awaiting an image</h2>
          <p>Your prediction and recommended disposal route will appear here.</p>
        </div> : <div className="result-content">
          <div className="route-card" style={{ '--route': route.color }}>
            <div className="route-orb"><span>{result.route === 'UNKNOWN' ? '!' : result.route[0]}</span></div>
            <div>
              <small>Recommended action</small>
              <h2>{route.label}</h2>
              <p>{route.detail}</p>
            </div>
          </div>

          <div className="prediction-head">
            <div><small>Detected item</small><h3>{result.predicted_label}</h3></div>
            <div className="confidence"><strong>{Math.round(result.confidence * 100)}%</strong><span>confidence</span></div>
          </div>

          <div className="bars">
            {result.top_predictions.slice(0, 4).map((item) => <div className="bar-row" key={item.type}>
              <div><span>{item.label}</span><b>{Math.round(item.probability * 100)}%</b></div>
              <div className="track"><i style={{ width: `${Math.max(2, item.probability * 100)}%` }} /></div>
            </div>)}
          </div>

          <div className={`safety ${result.held ? 'hold' : 'clear'}`}>
            <span>{result.held ? '!' : '✓'}</span>
            <div><strong>{result.held ? 'Safety hold applied' : 'Safety checks passed'}</strong>
              <p>{result.held ? (result.reason || 'Manual review is required.') : `Image confidence passed the local safety threshold (${Math.round(result.confidence * 100)}%).`}</p>
            </div>
          </div>
          <button className="reset" onClick={reset}>Analyze another item</button>
        </div>}
      </div>
    </section>

    <footer><span>Model runs entirely on this laptop</span><span>•</span><span>No image is uploaded to the cloud</span></footer>
  </main>
}

export default App
