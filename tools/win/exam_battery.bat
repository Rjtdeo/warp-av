cd /d C:\Users\Rajat\Desktop\warp-av
set PYTHONUTF8=1
set PY=C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe
del /q rl\exams\battery_done.txt 2>nul
rem Five harder exams on the current brain, 30 attempts each, same seed. Run INSTEAD of the trainer.
%PY% -u rl\eval_parking.py --episodes 30 --seed 2026 --tag far24 --lane-start 24 > rl\exams\battery_far24.txt 2>&1
%PY% -u rl\eval_parking.py --episodes 30 --seed 2026 --tag crooked15 --yaw-noise 15 > rl\exams\battery_crooked15.txt 2>&1
%PY% -u rl\eval_parking.py --episodes 30 --seed 2026 --tag offset1m --lateral-noise 1.0 > rl\exams\battery_offset1m.txt 2>&1
%PY% -u rl\eval_parking.py --episodes 30 --seed 2026 --tag noisyeyes --obs-noise 0.25,3,0.2 > rl\exams\battery_noisyeyes.txt 2>&1
%PY% -u rl\eval_parking.py --episodes 30 --seed 2026 --tag everything --lane-start 24 --yaw-noise 15 --lateral-noise 1.0 --obs-noise 0.25,3,0.2 > rl\exams\battery_everything.txt 2>&1
echo done > rl\exams\battery_done.txt
