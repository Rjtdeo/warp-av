# Bring this clone to exactly what GitHub has, without losing work that is not in git.
#
# Why a script: files are hand-copied onto this machine all day while something is being
# tested, and a plain `git pull` then ABORTS -- a copied file that a new commit also adds is
# an untracked file the pull would overwrite. That happened on 2026-09-11 and a whole
# validation run was driven by the OLD code without anyone noticing. So: revert tracked
# files, delete only the untracked files that the incoming commits bring anyway, and pull.
#
# Anything not in git and not in the incoming commits -- the RL recordings, the scratch
# folder, the logs -- is never touched.
Set-Location C:\Users\Rajat\Desktop\warp-av
git fetch -q origin
git checkout -- .
$incoming = git ls-tree -r --name-only origin/master
$untracked = git ls-files --others --exclude-standard
foreach ($f in $untracked) {
    if ($incoming -contains $f) {
        Remove-Item -Force (Join-Path (Get-Location) ($f -replace '/', '\'))
        "  removed the hand-copied $f (the pull brings it)"
    }
}
git pull --ff-only origin master 2>&1 | Select-String -NotMatch 'warning:'
"HEAD: " + (git rev-parse --short HEAD)
$left = git status --short | Select-String -NotMatch '^\?\? rl/|^\?\? tools/win/|^\?\? logs/|^\?\? scratch/'
"files still out of sync: " + ($left | Measure-Object).Count
