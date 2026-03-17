import { createFileRoute } from '@tanstack/react-router'
import "gchessboard";

export const Route = createFileRoute('/')({ component: App })

function App() {
  return (
    <g-chess-board fen="start" interactive></g-chess-board>
  )
}
