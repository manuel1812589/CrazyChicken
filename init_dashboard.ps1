# ================================
# Proyecto: Admin Dashboard - Dash
# Estructura en carpeta actual
# ================================

$basePath = Get-Location

Write-Host "Creando estructura en: $basePath"

# Estructura principal
$folders = @(
    "app",
    "app/auth",
    "app/db",
    "app/layouts",
    "app/pages",
    "app/components",
    "app/services",
    "assets",
    "assets/css",
    "assets/images",
    "logs",
    "config"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Path (Join-Path $basePath $folder) -Force | Out-Null
}

# Archivos principales
$files = @(
    "app/app.py",
    "app/server.py",
    "app/auth/login.py",
    "app/auth/session.py",
    "app/db/connection_auth.py",
    "app/db/connection_mart.py",
    "app/layouts/main_layout.py",
    "app/pages/login_page.py",
    "app/pages/dashboard_page.py",
    "app/pages/analytics_page.py",
    "app/components/navbar.py",
    "app/components/sidebar.py",
    "app/services/auth_service.py",
    "app/services/data_service.py",
    "config/settings.py",
    "config/secrets.example.py",
    "assets/css/styles.css",
    "requirements.txt",
    ".gitignore",
    "README.md"
)

foreach ($file in $files) {
    New-Item -ItemType File -Path (Join-Path $basePath $file) -Force | Out-Null
}

Write-Host "✅ Estructura creada correctamente en la carpeta actual"
Write-Host "👉 El entorno virtual (venv) irá aquí mismo, sin peleas"
