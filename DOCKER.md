# Skull King Docker Quick Start

## 🚀 Quick Start

### Development Mode (with hot reload)
```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f app

# Stop services
docker-compose -f docker-compose.dev.yml down
```

### Production Mode
```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

## 📦 Services

| Service | Port | Description |
|---------|------|-------------|
| FastAPI App | 8000 | Main application |
| MongoDB | 27017 | Database |
| Redis | 6379 | Cache & Pub/Sub |
| TensorBoard | 6006 | RL training visualization |

## 🔧 Configuration

Create a `.env` file (copy from `.env.example`):
```bash
cp .env.example .env
```

Key variables:
- `MONGODB_DATABASE` - Database name
- `JWT_SECRET` - Secret for JWT tokens
- `FRONTEND_URL` - Frontend application URL
- `ENABLE_BOTS` - Enable AI bots (true/false)

## 🎮 Using the Stack

### Access the API
```bash
# Health check
curl http://localhost:8000/health

# Get all cards
curl http://localhost:8000/games/cards

# API documentation
open http://localhost:8000/docs
```

### Run Bot Games
```bash
# Enter the container
docker-compose exec app bash

# Run bot game
python scripts/play_bot_game.py --players 4
```

### Train RL Agent
```bash
# Enter the container
docker-compose exec app bash

# Train agent
python scripts/train_rl_agent.py train --timesteps 1000000

# Monitor with TensorBoard
open http://localhost:6006
```

## 🛠️ Development

### Hot Reload
The development docker-compose mounts source code, so changes are reflected immediately.

### Run Tests
```bash
docker-compose exec app pytest
```

### Shell Access
```bash
docker-compose exec app bash
```

## 📊 Monitoring

### Application Logs
```bash
docker-compose logs -f app
```

### Database
```bash
# Connect to MongoDB
docker-compose exec mongodb mongosh skullking

# View collections
db.games.find().pretty()
```

### Redis
```bash
# Connect to Redis
docker-compose exec redis redis-cli

# Monitor commands
MONITOR
```

## 🔄 Updates

### Rebuild After Code Changes
```bash
# Production
docker-compose up -d --build app

# Development (not needed with hot reload)
docker-compose -f docker-compose.dev.yml restart app
```

### Update Dependencies
```bash
# Rebuild with new dependencies
docker-compose build --no-cache app
docker-compose up -d app
```

## 🧹 Cleanup

### Stop and Remove All
```bash
docker-compose down -v  # -v removes volumes
```

### Remove Images
```bash
docker rmi $(docker images 'skullking*' -q)
```

## 🎯 Useful Commands

```bash
# View running containers
docker-compose ps

# View resource usage
docker stats

# Execute command in container
docker-compose exec app python scripts/gym_example.py

# Copy files from container
docker cp skullking-app:/app/models/trained_agent.zip ./

# View environment variables
docker-compose exec app env | grep MONGODB
```

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9
```

### MongoDB Connection Issues
```bash
# Check if MongoDB is healthy
docker-compose exec mongodb mongosh --eval "db.adminCommand('ping')"

# Restart MongoDB
docker-compose restart mongodb
```

### Reset Everything
```bash
# Stop and remove all
docker-compose down -v

# Remove all Docker data
docker system prune -a --volumes
```

## 🌐 Production Deployment

### Environment Setup
1. Set strong `JWT_SECRET`
2. Configure MongoDB authentication
3. Set `ENVIRONMENT=production`
4. Use environment-specific `.env.production`

### Security
- Don't expose MongoDB/Redis ports publicly
- Use reverse proxy (nginx) for HTTPS
- Enable MongoDB authentication
- Set Redis password

### Scaling
```bash
# Scale app instances
docker-compose up -d --scale app=3
```
