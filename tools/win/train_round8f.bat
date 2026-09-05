cd /d C:\Users\Rajat\Desktop\warp-av
set PYTHONUTF8=1
rem Round 8f: as 8e plus "hold the lane while a car is beside you"; resume at rung 2 (car 3 bays back), it had 80% there.
C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe -u rl\train_parking.py --resume --start-level 5 --start-stage 1 --start-rung 1 --reverse --obstacles --lane-start 16 --explore-std 0.3 --steps 1400000 > rl\train_out.txt 2>&1
