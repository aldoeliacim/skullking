import { motion } from 'framer-motion';
import styles from './Button.module.css';

interface ButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'outline' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  fullWidth?: boolean;
  icon?: string;
  className?: string;
  type?: 'button' | 'submit';
}

export function Button({
  children,
  onClick,
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  fullWidth = false,
  icon,
  className = '',
  type = 'button',
}: ButtonProps) {
  return (
    <motion.button
      type={type}
      className={`${styles.button} ${styles[variant]} ${styles[size]} ${fullWidth ? styles.fullWidth : ''} ${className}`}
      onClick={onClick}
      disabled={disabled || loading}
      whileHover={disabled ? undefined : { scale: 1.02 }}
      whileTap={disabled ? undefined : { scale: 0.98 }}
    >
      {loading ? (
        <span className={styles.spinner} />
      ) : (
        <>
          {icon && <span className={styles.icon}>{icon}</span>}
          {children}
        </>
      )}
    </motion.button>
  );
}

export default Button;
