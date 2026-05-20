# week11 — conda 환경 (week10-dezero 공용) + gymnasium + DeZero 패치
#
#   cd week11
#   .\setup_conda.ps1
#   conda activate week10-dezero
#   python dqn2.py

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "conda 를 찾을 수 없습니다."
}

$condaExe = $env:CONDA_EXE
if (-not $condaExe) { $condaExe = (Get-Command conda.exe -ErrorAction Stop).Source }

$envName = "week10-dezero"
$yml = Join-Path $PSScriptRoot "environment.yml"

$exists = & $condaExe env list 2>$null | Select-String -Pattern "^\s*$([regex]::Escape($envName))\s"
if ($exists) {
    Write-Host "[1/3] conda env update" -ForegroundColor Cyan
    & $condaExe env update -f $yml --prune
} else {
    Write-Host "[1/3] conda env create" -ForegroundColor Cyan
    & $condaExe env create -f $yml
}
if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 3) { Write-Error "conda 실패 (exit $LASTEXITCODE)" }

Write-Host "`n[2/3] DeZero NumPy2 패치" -ForegroundColor Cyan
$patch = Join-Path $PSScriptRoot "..\week10\patch_dezero_numpy2.py"
if (-not (Test-Path $patch)) { $patch = Join-Path $PSScriptRoot "patch_dezero_numpy2.py" }
& $condaExe run -n $envName python $patch
if ($LASTEXITCODE -ne 0) { Write-Error "patch 실패" }

Write-Host "`n[3/3] import 확인" -ForegroundColor Cyan
& $condaExe run -n $envName python -c @"
import dezero, numpy as np
import gymnasium
print('dezero:', dezero.__file__)
print('numpy:', np.__version__)
print('gymnasium:', gymnasium.__version__)
"@

Write-Host @"

완료. dqn2.py 는 (base) 가 아니라 반드시:

  conda activate $envName
  cd $($PSScriptRoot)
  python Replay_buffer.py
  python dqn2.py

"@ -ForegroundColor Green
