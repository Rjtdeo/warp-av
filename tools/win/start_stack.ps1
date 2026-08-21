# Start the Warp AV stack detached from whoever called this (SSH-safe).
# A scheduled task survives the SSH session; Start-Process children do not.
schtasks /Delete /TN WarpAVStack /F 2>$null
schtasks /Create /F /TN WarpAVStack /SC ONCE /ST 00:00 /TR 'cmd /c "cd /d C:\Users\Rajat\Desktop\warp-av && C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe -u run.py > logs\stack_out.log 2>&1"' | Out-Null
schtasks /Run /TN WarpAVStack
