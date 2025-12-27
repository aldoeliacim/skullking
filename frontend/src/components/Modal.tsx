import { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import styles from './Modal.module.css';

interface ModalProps {
  isOpen: boolean;
  onClose?: () => void;
  children: React.ReactNode;
  title?: string;
  icon?: string;
  closable?: boolean;
}

export function Modal({
  isOpen,
  onClose,
  children,
  title,
  icon,
  closable = true,
}: ModalProps) {
  // Prevent body scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className={styles.overlay}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={closable ? onClose : undefined}
        >
          <motion.div
            className={styles.modal}
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            transition={{ type: 'spring', damping: 20, stiffness: 300 }}
            onClick={(e) => e.stopPropagation()}
          >
            {(icon || title) && (
              <div className={styles.header}>
                {icon && <span className={styles.icon}>{icon}</span>}
                {title && <h2 className={styles.title}>{title}</h2>}
                {closable && onClose && (
                  <button className={styles.closeButton} onClick={onClose}>
                    ✕
                  </button>
                )}
              </div>
            )}
            <div className={styles.content}>{children}</div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export default Modal;
