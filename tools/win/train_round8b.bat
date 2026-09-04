cd /d C:\Users\Rajat\Desktop\warp-av
set PYTHONUTF8=1
rem Round 8b: resume the round-8 brain; stages now judged on their own hazard only; hazard logged.
C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe -u rl\train_parking.py --resume --start-level 5 --start-stage 1 --reverse --obstacles --steps 1800000 > rl\train_out.txt 2>&1
