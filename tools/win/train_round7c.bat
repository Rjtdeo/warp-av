cd /d C:\Users\Rajat\Desktop\warp-av
set PYTHONUTF8=1
rem Round 7c (second half): RESUME with the stronger too-close warning (PROXIMITY_PAY 15).
C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe -u rl\train_parking.py --resume --start-level 5 --steps 1300000 > rl\train_out.txt 2>&1
