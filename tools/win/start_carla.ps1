# Start CARLA detached from the SSH session (see start_stack.ps1).
schtasks /Delete /TN WarpAVCarla /F 2>$null
schtasks /Create /F /TN WarpAVCarla /SC ONCE /ST 00:00 /TR "C:\CARLA\WindowsNoEditor\CarlaUE4.exe" | Out-Null
schtasks /Run /TN WarpAVCarla
