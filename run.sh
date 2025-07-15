#!/bin/bash

# ----- Prompt user for the data directory -----
read -p "Enter the path to your data directory (e.g., /home/user/data): " USER_INPUT

# Normalize and resolve absolute path
NORMALIZED_PATH=$(realpath "$USER_INPUT" 2>/dev/null)

# Validate input
if [ ! -d "$NORMALIZED_PATH" ]; then
    echo "The specified path does not exist or is not a directory. Exiting..."
    exit 1
fi

# ----- Docker information -----
CONTAINER_NAME="neuroglancer"
CONTAINER_WORKSPACE_PATH="/workspace"

HOST_DATA_PATH="$NORMALIZED_PATH"
CONTAINER_DATA_PATH="/workspace/datas"

HOST_UTILES_PATH="./utils"
CONTAINER_UTILES_PATH="/workspace/utils"

# ----- Docker Compose file path -----
COMPOSE_FILE="./docker-compose.yml"

echo "Generating docker-compose.yml..."
cat > "$COMPOSE_FILE" <<EOF
services:
  ${CONTAINER_NAME}:
    container_name: ${CONTAINER_NAME}
    build:
      context: ./
      dockerfile: Dockerfile
    volumes:
      - "${HOST_DATA_PATH}:${CONTAINER_DATA_PATH}"
      - "${HOST_UTILES_PATH}:${CONTAINER_UTILES_PATH}"
    working_dir: ${CONTAINER_WORKSPACE_PATH}
    command: bash
    stdin_open: true
    tty: true
EOF

# ----- Start the Docker container -----
echo "Starting container via docker-compose up --build..."
docker compose up --build

# ----- Clean up after exit -----
echo -e "\nStopping and cleaning up Docker container..."
docker compose down

if [ -f "$COMPOSE_FILE" ]; then
    rm -f "$COMPOSE_FILE"
    echo "Removed generated docker-compose.yml"
fi

echo "Pruning unused Docker images..."
docker image prune -f