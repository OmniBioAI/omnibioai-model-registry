import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@omnibioai/design-tokens/tokens.css'
import './index.css'
import App from './App.tsx'
import { connectAppearanceMessages } from './embedAppearance.js'

connectAppearanceMessages(window, document, import.meta.env.VITE_STUDIO_DEV_ORIGIN || '')

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
