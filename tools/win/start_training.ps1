# Overnight RL parking training as a scheduled task (SSH-safe detachment).
schtasks /Delete /TN WarpAVTrain /F 2>$null
schtasks /Create /F /TN WarpAVTrain /SC ONCE /ST 00:00 /TR "C:\Users\Rajat\Desktop\warp-av\tools\win\train_parking.bat" | Out-Null
schtasks /Run /TN WarpAVTrain
