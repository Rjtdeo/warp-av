# Overnight RL parking training as a scheduled task (SSH-safe detachment).
# Optional: pass the .bat to run; default is the fresh-training one.
param([string]$Bat = "C:\Users\Rajat\Desktop\warp-av\tools\win\train_parking.bat")
schtasks /Delete /TN WarpAVTrain /F 2>$null
schtasks /Create /F /TN WarpAVTrain /SC ONCE /ST 00:00 /TR "$Bat" | Out-Null
schtasks /Run /TN WarpAVTrain
