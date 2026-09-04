"""Lidar bays become planner slots in the world frame, sliced like map slots."""
import math

from warp_av.perception.bay_finder import Bay, SLOT_LEN_M, SLOT_WID_M
from warp_av.planning.sensed_slots import bays_to_slots, vehicle_to_world, slot_dict


def bay(x, y_left, yaw, start, end):
    mid = 0.5 * (start + end)
    return Bay(x=x, y=y_left, yaw=yaw, length=end - start, along_start=start, along_end=end,
               slot_start=mid - 3.5, slot_end=mid + 3.5)


def test_vehicle_to_world_puts_a_right_hand_bay_on_the_right():
    # ego at origin facing +x: 2 m to the RIGHT (y_left = -2) is world y = +2 in CARLA's y-right frame
    wx, wy = vehicle_to_world(5.0, -2.0, 0.0, 0.0, 0.0)
    assert abs(wx - 5.0) < 1e-9 and abs(wy - 2.0) < 1e-9
    # ego facing +y (yaw 90 deg, CARLA's yaw turns +x towards +y): 5 m ahead is +y,
    # and the van's right-hand side points to -x
    wx, wy = vehicle_to_world(5.0, -2.0, 10.0, 10.0, math.radians(90))
    assert abs(wx - (10.0 - 2.0)) < 1e-9 and abs(wy - 15.0) < 1e-9


def test_a_long_free_stretch_becomes_consecutive_slots():
    b = bay(x=10.0, y_left=-3.0, yaw=0.0, start=-2.0, end=23.0)     # 25 m free -> 3 slots of 7
    slots = bays_to_slots([b], 0.0, 0.0, 0.0)
    assert len(slots) == 3
    xs = sorted(sl["x"] for sl in slots)
    assert all(abs((xs[i + 1] - xs[i]) - SLOT_LEN_M) < 1e-6 for i in range(2))
    assert all(abs(sl["y"] - 3.0) < 1e-6 for sl in slots)               # right of the van
    assert all(sl["width"] == SLOT_WID_M and sl["length"] == SLOT_LEN_M for sl in slots)
    assert all(len(sl["corners"]) == 4 and sl["source"] == "lidar" for sl in slots)


def test_a_stretch_too_short_for_a_slot_gives_nothing():
    assert bays_to_slots([bay(3.0, -3.0, 0.0, 0.0, 7.5)], 0.0, 0.0, 0.0) == []


def test_slots_behind_the_van_are_dropped():
    b = bay(x=-14.0, y_left=-3.0, yaw=0.0, start=-26.0, end=-2.0)     # all behind
    assert bays_to_slots([b], 0.0, 0.0, 0.0) == []


def test_slot_heading_follows_the_kerb_and_the_ego():
    b = bay(x=10.0, y_left=-3.0, yaw=math.radians(5), start=6.0, end=14.0)
    slots = bays_to_slots([b], 0.0, 0.0, math.radians(30))
    assert len(slots) == 1 and abs(slots[0]["yaw"] - math.radians(35)) < 1e-3
