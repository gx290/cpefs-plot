[CmdletBinding()]
param(
    [ValidatePattern("^\d{14}$")]
    [string]$InputTime = "20260724000000",

    [switch]$Redraw
)

$ErrorActionPreference = "Stop"

$pythonArguments = @(
    (Join-Path $PSScriptRoot "batch_product_generator.py")
    "--input-time"
    $InputTime
    "--config"
    (Join-Path $PSScriptRoot "config.json")
)

if ($Redraw) {
    $pythonArguments += "--redraw"
}

& python @pythonArguments
exit $LASTEXITCODE
