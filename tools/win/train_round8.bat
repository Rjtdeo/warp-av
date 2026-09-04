cd /d C:\Users\Rajat\Desktop\warp-av
set PYTHONUTF8=1
set PY=C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe
rem Round 8: reverse gear + feelers + hazards in stages. Warm-started from round 6 (parking
rem intact, feelers and the gear born asleep), resumed at the full distance.
%PY% -u rl\warm_start.py --from rl\models\parking_ppo_round6.zip --reverse > rl\warm_out.txt 2>&1
%PY% -u rl\train_parking.py --resume --start-level 5 --reverse --obstacles --steps 2000000 > rl\train_out.txt 2>&1
