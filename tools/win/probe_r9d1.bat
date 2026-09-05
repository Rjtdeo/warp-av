cd /d C:\Users\Rajat\Desktop\warp-av
set PYTHONUTF8=1
rem Round 9: probe the brain after DAgger round 1 (empty / car two bays back / car right behind).
C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe -u rl\eval_parking.py --episodes 20 --seed 3 --reverse --model rl\models\parking_ppo_round9_bc.zip --tag r9d1_empty > rl\probe_out.txt 2>&1
C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe -u rl\eval_parking.py --episodes 15 --seed 3 --reverse --model rl\models\parking_ppo_round9_bc.zip --neighbour-p 1.0 --behind-bays 2 --tag r9d1_two_back >> rl\probe_out.txt 2>&1
C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe -u rl\eval_parking.py --episodes 15 --seed 3 --reverse --model rl\models\parking_ppo_round9_bc.zip --neighbour-p 1.0 --behind-bays 1 --tag r9d1_right_behind >> rl\probe_out.txt 2>&1
echo PROBES DONE >> rl\probe_out.txt
