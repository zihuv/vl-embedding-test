param(
    [string]$ServerName = "127.0.0.1",
    [int]$ServerPort = 7860,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Backend = "auto",
    [string]$OnnxProviders = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

$env:FGCLIP2_ORT_BACKEND = $Backend
if (-not $env:FGCLIP2_ORT_DLL_PATHS) {
    $dllCandidates = @(
        (Join-Path $env:USERPROFILE 'miniconda3\Lib\site-packages\torch\lib'),
        (Join-Path $env:USERPROFILE 'anaconda3\Lib\site-packages\torch\lib')
    )
    foreach ($candidate in $dllCandidates) {
        if (Test-Path -LiteralPath (Join-Path $candidate 'cudnn64_9.dll')) {
            $env:FGCLIP2_ORT_DLL_PATHS = $candidate
            break
        }
    }
}

if ($OnnxProviders) {
    $env:FGCLIP2_ORT_PROVIDERS = $OnnxProviders
    Write-Host "Launching app_compare_clip.py with explicit ORT providers: $OnnxProviders"
} else {
    Remove-Item Env:\FGCLIP2_ORT_PROVIDERS -ErrorAction SilentlyContinue
    Write-Host "Launching app_compare_clip.py with ORT backend profile: $Backend"
}
if ($env:FGCLIP2_ORT_DLL_PATHS) {
    Write-Host "Using CUDA DLL dirs: $env:FGCLIP2_ORT_DLL_PATHS"
}

& uv run --with onnxruntime-gpu python .\app_compare_clip.py `
    --fg-onnx-mode split-text `
    --server-name $ServerName `
    --server-port $ServerPort `
    @ExtraArgs

exit $LASTEXITCODE
