cd /d C:\Users\Rajat\Desktop\warp-av
set PYTHONUTF8=1
rem Round 8e: as 8d (approach car 4 -> 3 -> 2 bays back, action noise 0.3) plus the lane-hold charge: stay in the driving lane until 12 m before the bay.
C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe -u rl\train_parking.py --resume --start-level 5 --start-stage 1 --start-rung 0 --reverse --obstacles --lane-start 16 --explore-std 0.3 --steps 1600000 > rl\train_out.txt 2>&1
