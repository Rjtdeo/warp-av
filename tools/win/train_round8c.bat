cd /d C:\Users\Rajat\Desktop\warp-av
set PYTHONUTF8=1
rem Round 8c: resume the round-8 brain; far starts 22 m back so the car two bays back is passed, not born beside.
C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe -u rl\train_parking.py --resume --start-level 5 --start-stage 1 --reverse --obstacles --lane-start 22 --steps 1800000 > rl\train_out.txt 2>&1
