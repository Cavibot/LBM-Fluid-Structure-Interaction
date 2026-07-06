# Add .local\bin to User PATH if not already present
$target = "C:\Users\edwar\.local\bin"
$p = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($p -notlike "*$target*") {
    $newPath = $target + ";" + $p
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    Write-Output "Added '$target' to User PATH."
} else {
    Write-Output "'$target' is already in User PATH."
}
