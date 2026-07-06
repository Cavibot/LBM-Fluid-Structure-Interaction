$p = [Environment]::GetEnvironmentVariable("PATH", "User")
$parts = $p -split ";"
foreach ($part in $parts) {
    if ($part -like "*.local*" -or $part -like "*uv*") {
        Write-Output $part
    }
}
Write-Output "---"
Write-Output "Full User PATH:"
Write-Output $p
