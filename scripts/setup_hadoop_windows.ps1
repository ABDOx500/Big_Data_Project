# PowerShell script to set up Hadoop for Windows (required for PySpark)
# Run this script as Administrator

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Hadoop Windows Setup for PySpark 3.5.0   " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$hadoopHome = "C:\hadoop"
$hadoopBin = "$hadoopHome\bin"

# Create directories
Write-Host "[1/4] Creating Hadoop directories..." -ForegroundColor Yellow
if (-not (Test-Path $hadoopHome)) {
    New-Item -ItemType Directory -Path $hadoopHome -Force | Out-Null
    Write-Host "  Created: $hadoopHome" -ForegroundColor Green
}
if (-not (Test-Path $hadoopBin)) {
    New-Item -ItemType Directory -Path $hadoopBin -Force | Out-Null
    Write-Host "  Created: $hadoopBin" -ForegroundColor Green
}

# Download winutils and hadoop.dll
Write-Host ""
Write-Host "[2/4] Downloading Hadoop binaries for Windows..." -ForegroundColor Yellow

$winutilsUrl = "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.5/bin/winutils.exe"
$hadoopDllUrl = "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.5/bin/hadoop.dll"

try {
    # Download winutils.exe
    $winutilsPath = "$hadoopBin\winutils.exe"
    if (-not (Test-Path $winutilsPath)) {
        Write-Host "  Downloading winutils.exe..." -ForegroundColor Gray
        Invoke-WebRequest -Uri $winutilsUrl -OutFile $winutilsPath -UseBasicParsing
        Write-Host "  Downloaded: winutils.exe" -ForegroundColor Green
    } else {
        Write-Host "  winutils.exe already exists" -ForegroundColor Green
    }
    
    # Download hadoop.dll
    $hadoopDllPath = "$hadoopBin\hadoop.dll"
    if (-not (Test-Path $hadoopDllPath)) {
        Write-Host "  Downloading hadoop.dll..." -ForegroundColor Gray
        Invoke-WebRequest -Uri $hadoopDllUrl -OutFile $hadoopDllPath -UseBasicParsing
        Write-Host "  Downloaded: hadoop.dll" -ForegroundColor Green
    } else {
        Write-Host "  hadoop.dll already exists" -ForegroundColor Green
    }
} catch {
    Write-Host ""
    Write-Host "  ERROR: Failed to download files automatically." -ForegroundColor Red
    Write-Host "  Please download manually from:" -ForegroundColor Yellow
    Write-Host "  https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.5/bin" -ForegroundColor White
    Write-Host ""
    Write-Host "  Copy winutils.exe and hadoop.dll to: $hadoopBin" -ForegroundColor White
    Write-Host ""
}

# Set environment variables (User level)
Write-Host ""
Write-Host "[3/4] Setting environment variables..." -ForegroundColor Yellow

# Set HADOOP_HOME
[Environment]::SetEnvironmentVariable("HADOOP_HOME", $hadoopHome, "User")
Write-Host "  Set HADOOP_HOME = $hadoopHome" -ForegroundColor Green

# Add to PATH if not already present
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($currentPath -notlike "*$hadoopBin*") {
    $newPath = "$currentPath;$hadoopBin"
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    Write-Host "  Added $hadoopBin to PATH" -ForegroundColor Green
} else {
    Write-Host "  $hadoopBin already in PATH" -ForegroundColor Green
}

# Also set for current session
$env:HADOOP_HOME = $hadoopHome
$env:PATH = "$env:PATH;$hadoopBin"

# Verify installation
Write-Host ""
Write-Host "[4/4] Verifying installation..." -ForegroundColor Yellow

if (Test-Path "$hadoopBin\winutils.exe") {
    Write-Host "  winutils.exe: OK" -ForegroundColor Green
} else {
    Write-Host "  winutils.exe: MISSING" -ForegroundColor Red
}

if (Test-Path "$hadoopBin\hadoop.dll") {
    Write-Host "  hadoop.dll: OK" -ForegroundColor Green
} else {
    Write-Host "  hadoop.dll: MISSING" -ForegroundColor Red
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!                          " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "IMPORTANT: You must restart your terminal/IDE" -ForegroundColor Yellow
Write-Host "for the environment variables to take effect." -ForegroundColor Yellow
Write-Host ""
Write-Host "After restart, run your Flask app again:" -ForegroundColor White
Write-Host "  python web_app/app.py" -ForegroundColor Gray
Write-Host ""
