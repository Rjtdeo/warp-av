cd /d C:\Users\Rajat\Desktop\warp-av
set PYTHONUTF8=1
rem Round 7c: RESUME from the round-7b brain (parking intact, feelers half-learned).
rem Practice car now TWO bays back, never combined with a car ahead, and a
rem too-close warning through the feelers before any crash. Full distance from the start.
C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe -u rl\train_parking.py --resume --start-level 5 --steps 1400000 > rl\train_out.txt 2>&1
