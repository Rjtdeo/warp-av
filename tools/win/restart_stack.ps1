# Restart the Warp AV stack cleanly and wait until its API answers.
# `schtasks /End` alone does NOT stop it: it ends start_stack.bat, the python run.py child
# keeps running old code, and the next /Run fails quietly. So stop the python itself --
# matched by Name AND command line, because matching the command line alone also matches
# the PowerShell running this and kills your own SSH session. A stopped stack leaves its
# van and 9 sensors in CARLA, so clear those before and after.
Set-Location C:\Users\Rajat\Desktop\warp-av
$env:PYTHONPATH="C:\CARLA\WindowsNoEditor\PythonAPI\carla"
$py="C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe"
schtasks /End /TN WarpAVStack | Out-Null
Get-WmiObject Win32_Process | Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -like "*run.py*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep -Seconds 3
& $py tools\clear_leftover_vans.py --remove --sensors --stack-down 2>&1 | Select-String -NotMatch "WARNING|INFO"
schtasks /Run /TN WarpAVStack | Out-Null
for ($i = 0; $i -lt 40; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:5000/api/state | Out-Null; break } catch { Start-Sleep -Seconds 2 } }
Start-Sleep -Seconds 5
& $py tools\clear_leftover_vans.py --remove --sensors 2>&1 | Select-String -NotMatch "WARNING|INFO"
