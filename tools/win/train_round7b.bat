cd /d C:\Users\Rajat\Desktop\warp-av
set PYTHONUTF8=1
rem Round 7b: RESUME from the warm-started brain (round 6's skill + zeroed feeler
rem inputs, see rl\warm_start.py), starting on rung 3 (9 m) where cars ahead begin.
C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe -u rl\train_parking.py --resume --start-level 2 --steps 1800000 > rl\train_out.txt 2>&1
