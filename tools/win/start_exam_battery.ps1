# Harder-exam battery as a scheduled task (SSH-safe detachment), see start_training.ps1.
schtasks /Delete /TN WarpAVExam /F 2>$null
schtasks /Create /F /TN WarpAVExam /SC ONCE /ST 00:00 /TR "C:\Users\Rajat\Desktop\warp-av\tools\win\exam_battery.bat" | Out-Null
schtasks /Run /TN WarpAVExam
