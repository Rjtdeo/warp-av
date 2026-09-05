cd /d C:\Users\Rajat\Desktop\warp-av
set PYTHONUTF8=1
rem Round 9 step 5: practise from the copied-and-corrected brain, with a fading pull toward the instructor.
copy /Y rl\models\parking_ppo_round9_bc.zip rl\models\parking_ppo.zip
C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe -u rl\train_parking.py --resume --start-level 5 --start-stage 1 --start-rung 2 --max-stage 2 --lr 1.5e-4 --reverse --obstacles --lane-start 16 --lane-start-jitter 8 --teacher-weight 1.0 --teacher-fade 400000 --steps 600000 > rl\train_out.txt 2>&1
