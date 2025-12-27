# Skull King Frontend

A React 19 + Vite frontend for the Skull King card game.

## Tech Stack

- **React 19.2** with TypeScript
- **Vite 7** for development and bundling
- **Zustand 5** for state management
- **Framer Motion 12** for animations
- **react-i18next** for internationalization (English/Spanish)
- **CSS Modules** for component styling

## Development

```bash
# Install dependencies
npm install

# Start development server (http://localhost:5173)
npm run dev

# Type check
npm run typecheck

# Lint
npm run lint

# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
src/
├── components/       # Reusable UI components
│   ├── Card.tsx          # Card display with images
│   ├── Hand.tsx          # Player's hand
│   ├── TrickArea.tsx     # Current trick display
│   ├── Scoreboard.tsx    # Player scores
│   ├── Modal.tsx         # Base modal component
│   ├── BiddingModal.tsx  # Bid selection
│   ├── TigressModal.tsx  # Scary Mary choice
│   ├── AbilityModal.tsx  # Pirate abilities
│   ├── Button.tsx        # Styled buttons
│   └── Settings.tsx      # Language settings
├── pages/            # Screen components
│   ├── Home.tsx          # Welcome/join screen
│   ├── Lobby.tsx         # Game lobby
│   └── Game.tsx          # Main game screen
├── stores/           # State management
│   └── gameStore.ts      # Zustand store
├── services/         # API & WebSocket
│   ├── api.ts            # REST API client
│   └── websocket.ts      # WebSocket client
├── i18n/             # Translations
│   ├── en.json           # English
│   ├── es.json           # Spanish
│   └── index.ts          # i18n config
├── types/            # TypeScript types
│   └── game.ts           # Game-related types
├── utils/            # Helper functions
│   └── cardUtils.ts      # Card parsing/validation
└── styles/           # Global styles
    └── theme.css         # CSS variables
```

## Features

- **Responsive Design**: Works on desktop and mobile
- **Real-time Gameplay**: WebSocket connection with auto-reconnect
- **Pirate Abilities**: Full support for all 5 pirate abilities
- **Animations**: Smooth transitions with Framer Motion
- **i18n**: English and Spanish translations
- **Spectator Mode**: Watch games in progress

## Backend API

The frontend connects to:
- **REST API**: `http://localhost:8000/games/...`
- **WebSocket**: `ws://localhost:8000/games/ws/{game_id}/{player_id}`

Configure the backend URL by setting `VITE_API_URL` environment variable.
