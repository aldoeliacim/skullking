# Skull King Frontend

A React Native (Expo) mobile-first frontend for the Skull King card game.

## Tech Stack

- **Framework**: Expo SDK 52 with Expo Router
- **Language**: TypeScript (strict mode)
- **State Management**: Zustand
- **Animations**: React Native Reanimated
- **i18n**: i18next with react-i18next
- **Styling**: StyleSheet with custom theme system
- **Runtime**: Bun

## Prerequisites

- [Bun](https://bun.sh/) >= 1.0
- Node.js >= 18 (for Expo compatibility)
- iOS Simulator (for iOS development)
- Android Studio with emulator (for Android development)

## Getting Started

### Install dependencies

```bash
bun install
```

### Start development server

```bash
# Web
bun run web

# iOS
bun run ios

# Android
bun run android
```

### Linting and formatting

```bash
# Lint with Oxlint
bun run lint

# Format with Biome
bun run format

# Type check
bun run typecheck

# Run all checks
bun run check
```

## Project Structure

```
frontend/
├── app/                    # Expo Router screens
│   ├── _layout.tsx        # Root layout
│   ├── index.tsx          # Home/Login screen
│   ├── lobby/[id].tsx     # Game lobby
│   └── game/[id].tsx      # Game screen
├── src/
│   ├── components/        # Reusable UI components
│   │   ├── AbilityModal.tsx    # Pirate ability resolution UI
│   │   ├── BiddingModal.tsx    # Bid selection interface
│   │   ├── Button.tsx          # Themed button component
│   │   ├── Card.tsx            # Card display with animations
│   │   ├── Hand.tsx            # Player hand with play validation
│   │   ├── Input.tsx           # Themed text input
│   │   ├── Scoreboard.tsx      # Live scores with alliance indicators
│   │   ├── SettingsButton.tsx  # Language/sound settings
│   │   ├── TigressModal.tsx    # Scary Mary choice dialog
│   │   └── TrickArea.tsx       # Current trick display
│   ├── hooks/             # Custom React hooks
│   ├── i18n/              # Internationalization
│   │   ├── en.json
│   │   └── es.json
│   ├── services/          # API and WebSocket services
│   │   ├── api.ts
│   │   ├── websocket.ts
│   │   └── sounds.ts
│   ├── stores/            # Zustand state stores
│   │   └── gameStore.ts
│   └── styles/            # Theme and styling
│       └── theme.ts
├── assets/                # Images, fonts, sounds
├── app.json              # Expo config
├── package.json
├── tsconfig.json
├── oxlint.json           # Oxlint config
├── biome.json            # Biome formatter config
└── .pre-commit-config.yaml
```

## Features

- **Responsive Design**: Mobile-first with tablet/desktop support
- **Animations**: Smooth card dealing, playing, and scoring animations
- **Pirate Abilities**: Interactive modals for Rosie, Bendt, Harry, Jade, Roatan abilities
- **Loot Alliances**: Visual indicators showing player alliances and bonus tracking
- **Internationalization**: English and Spanish support
- **Real-time Updates**: WebSocket connection with automatic reconnection
- **Spectator Mode**: Watch games in progress
- **Bot Selection**: Choose AI difficulty (Easy/Medium/Hard/Neural Network)
- **Sound Effects**: Optional audio feedback

## Configuration

The app connects to the backend API. Configure the API URL:

1. Create a `.env` file:
   ```
   EXPO_PUBLIC_API_URL=http://localhost:8000
   ```

2. Or modify `src/services/api.ts` directly.

## Code Quality

- **Oxlint**: Fast linting with React and TypeScript rules
- **Biome**: Fast formatting
- **TypeScript**: Strict mode with comprehensive type checking
- **Pre-commit hooks**: Automated checks before commits

## Building for Production

### Web

```bash
bunx expo export --platform web
```

### iOS/Android

```bash
bunx eas build --platform ios
bunx eas build --platform android
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `bun run check` to ensure code quality
5. Submit a pull request

## License

MIT
