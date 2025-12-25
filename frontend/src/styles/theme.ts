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

// Color palette matching existing CSS
export const colors = {
  // Primary colors
  primary: '#e94560',
  primaryDark: '#c73e54',
  primaryLight: '#ff6b8a',

  // Background colors
  background: '#1a1a2e',
  backgroundLight: '#16213e',
  backgroundDark: '#0f0f1a',
  surface: '#1f1f3a',
  surfaceLight: '#2a2a4a',

  // Text colors
  text: '#eee',
  textMuted: '#aaa',
  textDark: '#888',

  // Accent colors
  accent: '#e94560',
  accentGold: '#f1c40f',
  accentBlue: '#3498db',
  accentGreen: '#27ae60',
  accentPurple: '#9b59b6',

  // Card suit colors
  suitBlue: '#4a90d9',
  suitYellow: '#f1c40f',
  suitGreen: '#27ae60',
  suitPurple: '#9b59b6',
  suitBlack: '#333',
  suitRed: '#e74c3c',

  // Status colors
  success: '#27ae60',
  warning: '#f39c12',
  error: '#e74c3c',
  info: '#3498db',

  // Overlay
  overlay: 'rgba(0, 0, 0, 0.7)',
  overlayLight: 'rgba(0, 0, 0, 0.5)',

  // Borders
  border: 'rgba(255, 255, 255, 0.1)',
  borderLight: 'rgba(255, 255, 255, 0.2)',
} as const;

// Typography
export const typography = {
  // Font families
  fontFamily: Platform.select({
    ios: 'System',
    android: 'Roboto',
    web: "'Segoe UI', system-ui, -apple-system, sans-serif",
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

// Shadows
export const shadows = {
  sm: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
    elevation: 2,
  },
  md: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 4,
  },
  lg: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  xl: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 16,
    elevation: 16,
  },
  glow: (color: string) => ({
    shadowColor: color,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.6,
    shadowRadius: 12,
    elevation: 12,
  }),
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
