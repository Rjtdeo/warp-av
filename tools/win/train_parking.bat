cd /d C:\Users\Rajat\Desktop\warp-av
set PYTHONUTF8=1
rem Round 5 starts FRESH. The round-3/4 checkpoint had learned to stand still
rem (10/10 parked when spawned already inside the box, 0/30 from the full
rem distance), so resuming would carry that habit forward. Keep the old brain
rem at rl\models\parking_ppo_round3.zip before running this.
C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe -u rl\train_parking.py --steps 1200000 > rl\train_out.txt 2>&1
