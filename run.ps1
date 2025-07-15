# ----- Prompt user for the data directory -----
$UserInput = Read-Host "Enter the path to your data directory (e.g., D:/path/to/data)"

# Normalize slashes (convert \ to /) and resolve to absolute path
$NormalizedPath = $UserInput -replace '\\', '/'
$ResolvedPath = Resolve-Path $NormalizedPath -ErrorAction SilentlyContinue

# Validate input
if (-not $ResolvedPath) {
    Write-Error "The specified path does not exist. Exiting..."
    exit 1
}

# ----- Docker information -----
$ContainerName = "zarr_analyzer"
$ContainerWorkspacePath = "/workspace"

$HostDataPath = $ResolvedPath.Path -replace '\\', '/'
$ContainerDataPath = "/workspace/datas"

$HostUtilesPath = "./utils"
$ContainerUtilesPath = "/workspace/utils"

# ----- Docker Compose file path -----
$ComposeFile = "./docker-compose.yml"

Write-Host "Generating docker-compose.yml..."
Set-Content -Path $ComposeFile -Value @"
services:
  ${ContainerName}:
    container_name: ${ContainerName}
    build:
      context: ./
      dockerfile: Dockerfile
    volumes:
      - "${HostDataPath}:${ContainerDataPath}"
      - "${HostUtilesPath}:${ContainerUtilesPath}"
    working_dir: ${ContainerWorkspacePath}
    command: bash
    stdin_open: true
    tty: true
"@

# ----- Start the Docker container ----- 
try {
    Write-Host "Starting container via docker-compose up --build..."
    docker compose up --build 
}
finally {
    Write-Host "Stopping and cleaning up Docker container..."
    docker compose down

    if (Test-Path $ComposeFile) {
        Remove-Item $ComposeFile -Force
        Write-Host "Removed generated docker-compose.yml"
    }

    Write-Host "Pruning unused Docker images..."
    docker image prune -f
}