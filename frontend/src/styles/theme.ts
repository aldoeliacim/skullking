import { Dimensions, Platform } from 'react-native';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// Responsive breakpoints
export const breakpoints = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
} as const;

// Check if we're on a small screen
export const isSmallScreen = SCREEN_WIDTH < breakpoints.sm;
export const isMediumScreen = SCREEN_WIDTH >= breakpoints.sm && SCREEN_WIDTH < breakpoints.md;
export const isLargeScreen = SCREEN_WIDTH >= breakpoints.md;

// Responsive scale factor
const guidelineBaseWidth = 375;
const guidelineBaseHeight = 812;

export const scale = (size: number): number => (SCREEN_WIDTH / guidelineBaseWidth) * size;
export const verticalScale = (size: number): number => (SCREEN_HEIGHT / guidelineBaseHeight) * size;
export const moderateScale = (size: number, factor = 0.5): number =>
  size + (scale(size) - size) * factor;

// Color palette inspired by Skull King box art
// Deep navy seas, ornate gold treasures, kraken teal, and vintage parchment
export const colors = {
  // Primary colors - Rich Gold (like the Skull King title)
  primary: '#d4a84b',
  primaryDark: '#b8923d',
  primaryLight: '#e6c36a',

  // Background colors - Deep Navy Blue (ocean depths)
  background: '#0a1628',
  backgroundLight: '#0f1e32',
  backgroundDark: '#060e18',
  surface: '#12253d',
  surfaceLight: '#1a3352',

  // Text colors - Vintage Cream/Parchment
  text: '#f0ebe3',
  textMuted: '#a8a090',
  textDark: '#6b6560',

  // Accent colors
  accent: '#d4a84b', // Gold
  accentGold: '#d4a84b',
  accentTeal: '#2d8a8a', // Kraken tentacles
  accentCrimson: '#8b2635', // Pirate red
  accentPurple: '#4a3b5c', // Mysterious purple

  // Suit colors - Vintage card aesthetic
  suitBlue: '#3a7ca5', // Parrot (blue) - ocean blue
  suitYellow: '#d4a84b', // Treasure chest (yellow) - gold
  suitGreen: '#2d6b4f', // Map (green) - sea green
  suitPurple: '#5c4a6e', // Kraken (purple) - deep purple
  suitBlack: '#1a1a1a', // Jolly Roger (black)
  suitRed: '#8b2635', // Pirate flag (red)

  // Special card colors
  skullKing: '#1a1a1a',
  pirate: '#8b2635',
  mermaid: '#2d8a8a',
  escape: '#f0ebe3',
  kraken: '#1a3352',
  whale: '#f0ebe3',

  // Status colors - Nautical themed
  success: '#2d8a8a', // Teal - safe waters
  warning: '#d4a84b', // Gold - treasure/caution
  error: '#8b2635', // Crimson - danger
  info: '#3a7ca5', // Ocean blue

  // Overlay - Deep sea darkness
  overlay: 'rgba(6, 14, 24, 0.85)',
  overlayLight: 'rgba(6, 14, 24, 0.6)',

  // Borders - Subtle gold tint
  border: 'rgba(212, 168, 75, 0.2)',
  borderLight: 'rgba(212, 168, 75, 0.35)',
  borderGold: 'rgba(212, 168, 75, 0.6)',
} as const;

// Typography - Nautical/Pirate themed
export const typography = {
  // Font families - Using Pirata One for titles on web (loaded via +html.tsx)
  fontFamily: Platform.select({
    ios: 'System',
    android: 'Roboto',
    web: "'Crimson Text', 'Georgia', serif",
  }),
  fontFamilyDisplay: Platform.select({
    ios: 'System',
    android: 'Roboto',
    web: "'Pirata One', 'Crimson Text', cursive",
  }),
  fontFamilyMono: Platform.select({
    ios: 'Menlo',
    android: 'monospace',
    web: "'Fira Code', 'Consolas', monospace",
  }),

  // Font sizes (responsive)
  fontSize: {
    xs: moderateScale(10),
    sm: moderateScale(12),
    base: moderateScale(14),
    md: moderateScale(16),
    lg: moderateScale(18),
    xl: moderateScale(20),
    '2xl': moderateScale(24),
    '3xl': moderateScale(30),
    '4xl': moderateScale(36),
    '5xl': moderateScale(48),
  },

  // Font weights
  fontWeight: {
    normal: '400' as const,
    medium: '500' as const,
    semibold: '600' as const,
    bold: '700' as const,
    extrabold: '800' as const,
  },

  // Line heights
  lineHeight: {
    tight: 1.2,
    normal: 1.5,
    relaxed: 1.75,
  },
} as const;

// Spacing (responsive)
export const spacing = {
  xs: moderateScale(4),
  sm: moderateScale(8),
  md: moderateScale(12),
  base: moderateScale(16),
  lg: moderateScale(20),
  xl: moderateScale(24),
  '2xl': moderateScale(32),
  '3xl': moderateScale(40),
  '4xl': moderateScale(48),
  '5xl': moderateScale(64),
} as const;

// Border radius
export const borderRadius = {
  none: 0,
  sm: moderateScale(4),
  base: moderateScale(8),
  md: moderateScale(12),
  lg: moderateScale(16),
  xl: moderateScale(24),
  full: 9999,
} as const;

// Shadows - Deep ocean shadows with golden accent glows
export const shadows = {
  sm: {
    shadowColor: '#060e18',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.4,
    shadowRadius: 2,
    elevation: 2,
  },
  md: {
    shadowColor: '#060e18',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.5,
    shadowRadius: 6,
    elevation: 4,
  },
  lg: {
    shadowColor: '#060e18',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.6,
    shadowRadius: 12,
    elevation: 8,
  },
  xl: {
    shadowColor: '#060e18',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.7,
    shadowRadius: 20,
    elevation: 16,
  },
  glow: (color: string) => ({
    shadowColor: color,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.7,
    shadowRadius: 16,
    elevation: 12,
  }),
  goldGlow: {
    shadowColor: '#d4a84b',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 12,
    elevation: 8,
  },
  tealGlow: {
    shadowColor: '#2d8a8a',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 12,
    elevation: 8,
  },
} as const;

// Animation durations
export const animation = {
  fast: 150,
  normal: 300,
  slow: 500,
  verySlow: 1000,
} as const;

// Z-index layers
export const zIndex = {
  base: 0,
  dropdown: 10,
  sticky: 20,
  overlay: 30,
  modal: 40,
  popover: 50,
  toast: 60,
} as const;

// Card dimensions (responsive) - sized for easy visibility and touch
export const cardDimensions = {
  width: moderateScale(90),
  height: moderateScale(126),
  widthLarge: moderateScale(120),
  heightLarge: moderateScale(168),
  widthSmall: moderateScale(70),
  heightSmall: moderateScale(98),
} as const;

// Screen dimensions
export const screen = {
  width: SCREEN_WIDTH,
  height: SCREEN_HEIGHT,
  isSmall: isSmallScreen,
  isMedium: isMediumScreen,
  isLarge: isLargeScreen,
} as const;

// Complete theme object
export const theme = {
  colors,
  typography,
  spacing,
  borderRadius,
  shadows,
  animation,
  zIndex,
  cardDimensions,
  screen,
  breakpoints,
  scale,
  verticalScale,
  moderateScale,
} as const;

export type Theme = typeof theme;
export type Colors = typeof colors;
export type Spacing = keyof typeof spacing;

export default theme;
