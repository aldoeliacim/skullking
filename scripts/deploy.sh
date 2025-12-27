#!/bin/bash
#
# Skull King Deployment Script
#
# Deploys frontend to PVE nginx and backend as systemd service.
# Usage: ./scripts/deploy.sh [frontend|backend|all]
#

set -euo pipefail

# Configuration
PVE_HOST="pve"
PVE_WEB_ROOT="/var/www/html/skullking.aldo.pw"
BACKEND_IP="192.168.1.8"
BACKEND_PORT="8000"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Build frontend using npm (Vite + React)
build_frontend() {
    log_info "Building frontend..."
    cd "$PROJECT_DIR/frontend"

    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        log_info "Installing dependencies..."
        npm install
    fi

    # Build for production
    log_info "Running Vite build..."
    npm run build

    log_success "Frontend built successfully"
}

# Deploy frontend to PVE
deploy_frontend() {
    log_info "Deploying frontend to PVE..."

    # Build first
    build_frontend

    # Create directory on PVE if it doesn't exist
    ssh "$PVE_HOST" "sudo mkdir -p $PVE_WEB_ROOT && sudo chown -R www-data:www-data $PVE_WEB_ROOT"

    # Rsync built files
    log_info "Syncing files to PVE..."
    rsync -avz --delete \
        "$PROJECT_DIR/frontend/dist/" \
        "$PVE_HOST:$PVE_WEB_ROOT/"

    # Fix permissions
    ssh "$PVE_HOST" "sudo chown -R www-data:www-data $PVE_WEB_ROOT"

    log_success "Frontend deployed to $PVE_HOST:$PVE_WEB_ROOT"
}

# Update PVE nginx config
update_nginx_config() {
    log_info "Updating PVE nginx config..."

    # Create the new config
    local nginx_config=$(cat <<'NGINX_EOF'
server {
    include /etc/nginx/snippets/listen-ssl.conf;

    server_name skullking.aldo.pw;

    include /etc/nginx/snippets/ssl-wildcard-aldo.conf;

    root /var/www/html/skullking.aldo.pw;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/json application/xml;

    # Backend API and WebSocket - proxy to backend
    location /games {
        include /etc/nginx/snippets/proxy_common.conf;
        proxy_pass http://BACKEND_IP:BACKEND_PORT/games;
        proxy_read_timeout 86400s;
    }

    # Backend health check
    location /health {
        include /etc/nginx/snippets/proxy_common.conf;
        proxy_pass http://BACKEND_IP:BACKEND_PORT/health;
    }

    # Card images from backend
    location /static/ {
        include /etc/nginx/snippets/proxy_common.conf;
        proxy_pass http://BACKEND_IP:BACKEND_PORT/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # JS/CSS bundles with hashes - cache aggressively
    location ~* \.[a-f0-9]+\.(js|css)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # HTML files - never cache (for SPA updates)
    location ~* \.html$ {
        expires -1;
        add_header Cache-Control "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0";
        add_header Pragma "no-cache";
    }

    # SPA fallback - serve index.html with no-cache
    location / {
        try_files $uri $uri/ /index.html;
        expires -1;
        add_header Cache-Control "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0";
        add_header Pragma "no-cache";
    }
}
NGINX_EOF
)

    # Replace placeholders
    nginx_config="${nginx_config//BACKEND_IP/$BACKEND_IP}"
    nginx_config="${nginx_config//BACKEND_PORT/$BACKEND_PORT}"

    # Write config to PVE
    echo "$nginx_config" | ssh "$PVE_HOST" "sudo tee /etc/nginx/sites-available/skullking > /dev/null"

    # Enable site if not already
    ssh "$PVE_HOST" "sudo ln -sf /etc/nginx/sites-available/skullking /etc/nginx/sites-enabled/skullking 2>/dev/null || true"

    # Test nginx config
    log_info "Testing nginx configuration..."
    if ssh "$PVE_HOST" "sudo nginx -t"; then
        log_success "Nginx config is valid"
    else
        log_error "Nginx config test failed!"
        return 1
    fi

    # Reload nginx
    log_info "Reloading nginx..."
    ssh "$PVE_HOST" "sudo systemctl reload nginx"

    log_success "PVE nginx config updated and reloaded"
}

# Deploy backend (start/restart uvicorn)
deploy_backend() {
    log_info "Deploying backend..."
    cd "$PROJECT_DIR"

    # Sync dependencies
    log_info "Syncing Python dependencies..."
    uv sync

    # Check if backend is already running
    if pgrep -f "uvicorn app.main:app" > /dev/null; then
        log_info "Stopping existing backend..."
        pkill -f "uvicorn app.main:app" || true
        sleep 2
    fi

    # Start backend in background
    log_info "Starting backend..."
    nohup uv run uvicorn app.main:app --host 0.0.0.0 --port "$BACKEND_PORT" > /tmp/skullking-backend.log 2>&1 &

    # Wait for health check
    log_info "Waiting for backend to be healthy..."
    for i in {1..30}; do
        if curl -s "http://localhost:$BACKEND_PORT/health" > /dev/null 2>&1; then
            log_success "Backend is healthy"
            return 0
        fi
        sleep 1
    done

    log_warn "Backend health check timeout (may still be starting)"
    log_info "Check logs: tail -f /tmp/skullking-backend.log"
}

# Full deployment
deploy_all() {
    log_info "Starting full deployment..."

    deploy_backend
    deploy_frontend
    update_nginx_config

    log_success "Full deployment complete!"
    echo ""
    echo "  Frontend: https://skullking.aldo.pw"
    echo "  Backend:  http://$BACKEND_IP:$BACKEND_PORT"
    echo ""
}

# Show backend status and logs
status() {
    log_info "Backend status..."
    if pgrep -f "uvicorn app.main:app" > /dev/null; then
        log_success "Backend is running"
        ps aux | grep "[u]vicorn app.main:app"
    else
        log_warn "Backend is not running"
    fi
    echo ""
    log_info "Recent logs:"
    tail -20 /tmp/skullking-backend.log 2>/dev/null || log_warn "No logs found"
}

# Stop backend
stop_backend() {
    log_info "Stopping backend..."
    if pkill -f "uvicorn app.main:app"; then
        log_success "Backend stopped"
    else
        log_warn "Backend was not running"
    fi
}

# Main
main() {
    local target="${1:-all}"

    case "$target" in
        frontend)
            deploy_frontend
            update_nginx_config
            ;;
        backend)
            deploy_backend
            ;;
        nginx)
            update_nginx_config
            ;;
        status)
            status
            ;;
        stop)
            stop_backend
            ;;
        all)
            deploy_all
            ;;
        *)
            echo "Usage: $0 [frontend|backend|nginx|status|stop|all]"
            echo ""
            echo "  frontend  - Build and deploy frontend to PVE"
            echo "  backend   - Start/restart backend uvicorn server"
            echo "  nginx     - Update PVE nginx config only"
            echo "  status    - Show backend status and logs"
            echo "  stop      - Stop backend server"
            echo "  all       - Deploy everything (default)"
            exit 1
            ;;
    esac
}

main "$@"
