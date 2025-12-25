# Build stage - use Node.js for Expo compatibility
FROM node:22-alpine AS builder

WORKDIR /app

# Install bun for faster package installation
RUN npm install -g bun

# Copy package files
COPY package.json bun.lock* ./

# Install dependencies with bun
RUN bun install --frozen-lockfile

# Copy source code
COPY . .

# Build Expo web app using npx (more stable than bunx for Metro)
RUN npx expo export --platform web

# Production stage
FROM nginx:alpine

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
