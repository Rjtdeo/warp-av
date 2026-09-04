cd /d C:\Users\Rajat\Desktop\warp-av
set PYTHONUTF8=1
rem Round 7 trains FRESH: the student now sees 9 numbers (5 pose + 4 feelers),
rem so no earlier brain fits. Real parked cars appear in the neighbouring bays
rem during practice. Copy the previous brain aside (rl\models\parking_ppo_roundN.zip)
rem before running this. 2M steps is about 9.5 h at the measured 59 steps/s.
C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe -u rl\train_parking.py --steps 2000000 > rl\train_out.txt 2>&1
