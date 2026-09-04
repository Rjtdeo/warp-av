cd /d C:\Users\Rajat\Desktop\warp-av
set PYTHONUTF8=1
rem Round 8d: resume the round-8 brain; approach car starts 4 bays back (36 m start) and walks closer; a little action noise for the new skill.
C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe -u rl\train_parking.py --resume --start-level 5 --start-stage 1 --start-rung 0 --reverse --obstacles --lane-start 16 --explore-std 0.3 --steps 1700000 > rl\train_out.txt 2>&1
