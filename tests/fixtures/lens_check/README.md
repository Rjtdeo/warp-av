# Clean-lens clips (Town10HD, CARLA 0.9.15, 2026-09-10)

Two runs of 12 consecutive thumbnails (every eighth pixel of the 800 x 600 front camera, the
picture the lens check itself uses) from the van's front camera, lens CLEAN, van driving:
`clear_noon` at 8 m/s and `wet_noon` at 2-3 m/s. On both, the old "something on the lens"
check called the camera broken -- 6 to 21 patches of plain sky "not changing" -- which slowed
the van to walking pace and then let the safety supervisor stop it.

Recorded with `tools/record_lens_check.py`; `tests/test_lens_check_real_frames.py` uses them.
