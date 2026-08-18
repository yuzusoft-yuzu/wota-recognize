# ============================================================
# Wota艺 动作识别系统 - Windows Server 部署脚本 (PowerShell)
# 用法: 以管理员身份运行 PowerShell, 执行:
#   powershell -ExecutionPolicy Bypass -File deploy_windows.ps1
# 可选环境变量: ADMIN_PASSWORD (默认 admin123), PORT (默认 5000)
# ============================================================
$ErrorActionPreference = "Stop"
$adminPass = if ($env:ADMIN_PASSWORD) { $env:ADMIN_PASSWORD } else { "admin123" }
$port = if ($env:PORT) { $env:PORT } else { "5000" }
$repoUrl = "https://github.com/yuzusoft-yuzu/wota-recognize.git"
$appDir = "C:\wota-recognize"
$pyVer = "3.11"

Write-Host "==> [1/6] 检查 Python 3.11" -ForegroundColor Cyan
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Host "未安装 Python，请先安装: https://www.python.org/downloads/release/python-3119/"
    Write-Host "安装时务必勾选 'Add python.exe to PATH'"
    exit 1
}
python --version

Write-Host "==> [2/6] 拉取代码" -ForegroundColor Cyan
if (Test-Path $appDir) { Remove-Item $appDir -Recurse -Force }
git clone $repoUrl $appDir
Set-Location $appDir

Write-Host "==> [3/6] 安装 Python 依赖 (含 mediapipe, 约 2-5 分钟)" -ForegroundColor Cyan
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "==> [4/6] 创建数据目录" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "$appDir\uploads" | Out-Null
New-Item -ItemType Directory -Force -Path "$appDir\static\output" | Out-Null

Write-Host "==> [5/6] 启动服务 (后台, 端口 $port)" -ForegroundColor Cyan
$env:ADMIN_USER = "admin"
$env:ADMIN_PASSWORD = $adminPass
$env:PORT = $port
# 杀掉旧的 wota 进程
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'waitress|app:app' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Process -FilePath "waitress-serve" -ArgumentList "--listen=0.0.0.0:$port", "--threads=8", "app:app" `
    -WorkingDirectory $appDir -WindowStyle Hidden
Start-Sleep -Seconds 10

Write-Host "==> [6/6] 验证服务" -ForegroundColor Cyan
$health = try { (Invoke-RestMethod "http://127.0.0.1:$port/api/health" -TimeoutSec 10) | ConvertTo-Json -Compress } catch { "{}" }
Write-Host "健康检查: $health"
Write-Host ""
Write-Host "============================================================"
Write-Host " 部署完成！"
Write-Host "   本地验证:   curl http://127.0.0.1:$port/api/health"
Write-Host "   外网访问:   http://<服务器公网IP>:$port"
Write-Host "   管理员:     admin / $adminPass"
Write-Host "   服务目录:   $appDir"
Write-Host ""
Write-Host " 若 health 返回 mediapipe_available:true 即为骨光融合模式"
Write-Host " 若外网不通，请检查阿里云安全组是否放行 TCP $port"
Write-Host "============================================================"
