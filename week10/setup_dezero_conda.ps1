# week10 Conda 환경 설치 (최초 1회 또는 갱신)
#
#   cd week10
#   .\setup_dezero_conda.ps1
#   conda activate week10-dezero
#   python dezero3.py
#   python quiz_q1_optimizer_compare.py
#   ...

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "conda 를 찾을 수 없습니다. Anaconda Prompt 에서 실행하세요."
}

$condaExe = $env:CONDA_EXE
if (-not $condaExe) {
    $condaExe = (Get-Command conda.exe -ErrorAction Stop).Source
}

$envName = "week10-dezero"
$yml = Join-Path $PSScriptRoot "environment.yml"

$exists = & $condaExe env list 2>$null | Select-String -Pattern "^\s*$([regex]::Escape($envName))\s"
if ($exists) {
    Write-Host "[1/2] 환경 갱신: conda env update -f environment.yml --prune" -ForegroundColor Cyan
    & $condaExe env update -f $yml --prune
}
else {
    Write-Host "[1/2] 환경 생성: conda env create -f environment.yml" -ForegroundColor Cyan
    & $condaExe env create -f $yml
}

$code = $LASTEXITCODE
if ($code -ne 0 -and $code -ne 3) {
    Write-Error "conda 실패 (exit $code)"
}

Write-Host "`n[2/2] import 확인" -ForegroundColor Cyan
& $condaExe run -n $envName python -c "import dezero, numpy as np; print('dezero:', dezero.__file__); print('numpy:', np.__version__)"

Write-Host @"

완료. 이제 아래 순서로 실행하세요:

  conda activate $envName
  cd $PSScriptRoot
  python dezero3.py
  python dezero4.py
  python q_learning_nn.py
  python quiz_q1_optimizer_compare.py
  python quiz_q2_sin_4pi.py
  python quiz_q3_gridworld.py

한 번에 돌리려면 (활성화 후): .\run_week10.ps1

"@ -ForegroundColor Green
