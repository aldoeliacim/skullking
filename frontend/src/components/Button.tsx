import React from 'react';
import {
  ActivityIndicator,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  type TextStyle,
  type ViewStyle,
} from 'react-native';
import { borderRadius, colors, shadows, spacing, typography } from '../styles/theme';

type ButtonVariant = 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';
type ButtonSize = 'sm' | 'md' | 'lg';

interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: ButtonVariant;
  size?: ButtonSize;
  disabled?: boolean;
  loading?: boolean;
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
  fullWidth?: boolean;
  style?: ViewStyle;
  textStyle?: TextStyle;
}

export function Button({
  title,
  onPress,
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  icon,
  iconPosition = 'left',
  fullWidth = false,
  style,
  textStyle,
}: ButtonProps): React.JSX.Element {
  const isDisabled = disabled || loading;

  const variantStyles = {
    primary: {
      container: styles.primaryContainer,
      text: styles.primaryText,
    },
    secondary: {
      container: styles.secondaryContainer,
      text: styles.secondaryText,
    },
    outline: {
      container: styles.outlineContainer,
      text: styles.outlineText,
    },
    ghost: {
      container: styles.ghostContainer,
      text: styles.ghostText,
    },
    danger: {
      container: styles.dangerContainer,
      text: styles.dangerText,
    },
  };

  const sizeStyles = {
    sm: {
      container: styles.smContainer,
      text: styles.smText,
    },
    md: {
      container: styles.mdContainer,
      text: styles.mdText,
    },
    lg: {
      container: styles.lgContainer,
      text: styles.lgText,
    },
  };

  return (
    <Pressable
      onPress={onPress}
      disabled={isDisabled}
      style={({ pressed }) => [
        styles.container,
        variantStyles[variant].container,
        sizeStyles[size].container,
        fullWidth && styles.fullWidth,
        isDisabled && styles.disabled,
        pressed && Platform.OS !== 'web' && styles.pressed,
        style,
      ]}
    >
      {loading ? (
        <ActivityIndicator
          color={variant === 'outline' || variant === 'ghost' ? colors.primary : colors.text}
          size="small"
        />
      ) : (
        <>
          {icon && iconPosition === 'left' && icon}
          <Text
            style={[
              styles.text,
              variantStyles[variant].text,
              sizeStyles[size].text,
              icon && iconPosition === 'left' ? styles.textWithLeftIcon : undefined,
              icon && iconPosition === 'right' ? styles.textWithRightIcon : undefined,
              textStyle,
            ]}
          >
            {title}
          </Text>
          {icon && iconPosition === 'right' && icon}
        </>
      )}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: borderRadius.base,
    ...shadows.sm,
    cursor: 'pointer' as unknown as undefined, // Web cursor style
  },
  fullWidth: {
    width: '100%',
  },
  disabled: {
    opacity: 0.5,
    cursor: 'not-allowed' as unknown as undefined,
  },
  pressed: {
    transform: [{ scale: 0.97 }],
    opacity: 0.9,
  },
  text: {
    fontWeight: typography.fontWeight.semibold,
  },
  textWithLeftIcon: {
    marginLeft: spacing.sm,
  },
  textWithRightIcon: {
    marginRight: spacing.sm,
  },

  // Variants
  primaryContainer: {
    backgroundColor: colors.primary,
    borderWidth: 1,
    borderColor: colors.primaryDark,
  },
  primaryText: {
    color: '#0a1628', // Dark navy for contrast on gold
  },
  secondaryContainer: {
    backgroundColor: colors.surfaceLight,
  },
  secondaryText: {
    color: colors.text,
  },
  outlineContainer: {
    backgroundColor: 'transparent',
    borderWidth: 2,
    borderColor: colors.primary,
  },
  outlineText: {
    color: colors.primary,
  },
  ghostContainer: {
    backgroundColor: 'transparent',
  },
  ghostText: {
    color: colors.primary,
  },
  dangerContainer: {
    backgroundColor: colors.error,
  },
  dangerText: {
    color: colors.text,
  },

  // Sizes
  smContainer: {
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.md,
  },
  smText: {
    fontSize: typography.fontSize.sm,
  },
  mdContainer: {
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
  },
  mdText: {
    fontSize: typography.fontSize.base,
  },
  lgContainer: {
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.xl,
  },
  lgText: {
    fontSize: typography.fontSize.lg,
  },
});

export default Button;
