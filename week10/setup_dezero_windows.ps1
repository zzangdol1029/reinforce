# week10 Windows용 DeZero 실습 환경 (가상환경 + 설치 + NumPy 2 패치)
# 관리자 권한 불필요 — week10 폴더 안에만 venv 생성
#
# 실행 (PowerShell):
#   cd week10
#   Set-ExecutionPolicy -Scope CurrentUser Bypass -Force   # 필요 시 한 번만
#   .\setup_dezero_windows.ps1
#
param(
    [string]$VenvDir = ".venv-week10"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Py = Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $Py) {
    Write-Host "python 을 찾을 수 없습니다. Anaconda Prompt 에서 conda activate 후 다시 실행하세요." -ForegroundColor Red
    exit 1
}

Write-Host "Python: $Py"
& $Py -m venv $VenvDir
if (-not (Test-Path "$VenvDir\Scripts\python.exe")) {
    Write-Host "venv 생성 실패" -ForegroundColor Red
    exit 1
}

$Vp = Resolve-Path "$VenvDir\Scripts\python.exe"
Write-Host "`nvenv: $Vp"

& $Vp -m pip install --upgrade pip setuptools wheel

& $Vp -m pip install "matplotlib>=3.5" "dezero>=0.0.13"

Write-Host "`nNumPy 2 호환 패치 적용 중 (dezero np.int 제거)"
& $Vp "$PSScriptRoot\patch_dezero_numpy2.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n검증: import dezero"
& $Vp -c "import dezero; print('dezero OK', dezero.__file__)"

Write-Host @"

완료.
이번 터미널에서만 활성화:
  .$VenvDir\Scripts\Activate.ps1

이후 실행 예:
  python quiz_q1_optimizer_compare.py

"@ -ForegroundColor Green
