cd /d C:\Users\Rajat\Desktop\warp-av
set PYTHONUTF8=1
rem Trains FRESH every round. Round 3's brain had learned to stand still,
rem round 5's to drive up to the bay and hover; resuming would carry the habit
rem forward. Copy the previous brain aside (rl\models\parking_ppo_roundN.zip)
rem before running this.
C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe -u rl\train_parking.py --steps 1200000 > rl\train_out.txt 2>&1
