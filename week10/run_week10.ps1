# week10 실습 스크립트 실행 (conda activate week10-dezero 후)
#
#   conda activate week10-dezero
#   cd week10
#   .\run_week10.ps1              # 전체
#   .\run_week10.ps1 -Only q1     # quiz_q1 만

param(
    [ValidateSet("all", "dezero3", "dezero4", "qlearning", "q1", "q2", "q3")]
    [string]$Only = "all"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($env:CONDA_DEFAULT_ENV -ne "week10-dezero") {
    Write-Host "먼저 환경을 활성화하세요:  conda activate week10-dezero" -ForegroundColor Yellow
    Write-Host "  (현재: $($env:CONDA_DEFAULT_ENV ?? '(base/없음)'))" -ForegroundColor DarkGray
    exit 1
}

$scripts = @{
    dezero3  = "dezero3.py"
    dezero4  = "dezero4.py"
    qlearning = "q_learning_nn.py"
    q1       = "quiz_q1_optimizer_compare.py"
    q2       = "quiz_q2_sin_4pi.py"
    q3       = "quiz_q3_gridworld.py"
}

$order = if ($Only -eq "all") {
    @("dezero3", "dezero4", "qlearning", "q1", "q2", "q3")
} else {
    @($Only)
}

foreach ($key in $order) {
    $file = $scripts[$key]
    Write-Host "`n========== $file ==========" -ForegroundColor Cyan
    python $file
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$file 실패 (exit $LASTEXITCODE)"
    }
}

Write-Host "`n모든 스크립트 실행 완료." -ForegroundColor Green
