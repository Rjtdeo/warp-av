cd /d C:\Users\Rajat\Desktop\warp-av
set PYTHONUTF8=1
rem Round 8h: round 8g again from the 19:02 brain with snapshots; never unlocks the unsolvable "car right behind" stage; half learning rate.
C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe -u rl\train_parking.py --resume --start-level 5 --start-stage 1 --start-rung 1 --max-stage 1 --lr 1.5e-4 --reverse --obstacles --lane-start 16 --lane-start-jitter 8 --explore-std 0.3 --steps 700000 > rl\train_out.txt 2>&1
