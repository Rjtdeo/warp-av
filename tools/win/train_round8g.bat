cd /d C:\Users\Rajat\Desktop\warp-av
set PYTHONUTF8=1
rem Round 8g: as 8f plus far starts spread over +0..8 m (16-24 m empty, 22-30 m with a car two bays back, ...): the lane lesson everywhere along the approach.
C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe -u rl\train_parking.py --resume --start-level 5 --start-stage 1 --start-rung 1 --reverse --obstacles --lane-start 16 --lane-start-jitter 8 --explore-std 0.3 --steps 1300000 > rl\train_out.txt 2>&1
