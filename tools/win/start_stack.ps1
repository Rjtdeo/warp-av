# Start the Warp AV stack detached from whoever called this (SSH-safe).
# A scheduled task survives the SSH session; Start-Process children do not.
schtasks /Delete /TN WarpAVStack /F 2>$null
schtasks /Create /F /TN WarpAVStack /SC ONCE /ST 00:00 /TR "C:\Users\Rajat\Desktop\warp-av\tools\win\start_stack.bat" | Out-Null
schtasks /Run /TN WarpAVStack
