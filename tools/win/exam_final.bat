cd /d C:\Users\Rajat\Desktop\warp-av
set PYTHONUTF8=1
set PY=C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe
del /q rl\exams\final_done.txt 2>nul
rem HARDER: everything doubled, new bays. Pass mark set in advance: 90%+ parked, zero collisions.
%PY% -u rl\eval_parking.py --episodes 60 --seed 777 --tag harder --lane-start 32 --yaw-noise 25 --lateral-noise 1.5 --obs-noise 0.5,6,0.4 > rl\exams\final_harder.txt 2>&1
rem HARDEST: built to find the edge. 40 m, 40 deg, 2 m off, noise x4, eyes freeze 1 step in 5, everything seen 0.3 s late.
%PY% -u rl\eval_parking.py --episodes 100 --seed 4242 --tag hardest --lane-start 40 --yaw-noise 40 --lateral-noise 2.0 --obs-noise 1.0,10,0.6 --obs-dropout 0.2 --obs-delay 3 > rl\exams\final_hardest.txt 2>&1
echo done > rl\exams\final_done.txt
