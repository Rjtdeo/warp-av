"""Reading the traffic light's colour off the camera instead of asking the simulator.

Phase 15 got the van to the light in time. It still asked CARLA what colour the light was,
which is not a question a real van can ask. This is the swap, and these tests hold the line
that made it safe to make: A COLOUR IS ONLY EVER REPORTED BY ITS OWN LAMP. The top third of
the lens column can say red and nothing else; the bottom third can say green and nothing else.
So no building, sign or tree in the picture can turn a red light green.

The numbers here are not invented. They are the pixel values measured off Town10HD: a lit red
lens reads 255/166/98, a lit amber 252/251/158, a lit green 194/255/151, and -- the one that
broke the first two attempts -- an UNLIT amber lens in daylight reads 220/187/0, which is more
colourful than any lit green lamp.
"""
import numpy as np
import pytest

from warp_av.perception.light_camera import (
    CameraLightReader, LampHead, LampMap, read_lamp_colour,
    LENS_LEFT, LENS_RIGHT, LENS_BOTTOM, MIN_LIT_GREEN, MIN_LENS_ROWS,
    AGREEMENT_MAX_AGE_S)
from warp_av.perception.traffic_lights import (
    GREEN, RED, UNKNOWN, YELLOW, SignalGeometry, SignalMap, TrafficLightLookahead)


# ---------------------------------------------------------------- a lamp head, in pixels

# measured on Town10HD, BGR as the camera delivers it
LIT_RED = (98, 166, 255)
LIT_AMBER = (158, 251, 252)
LIT_GREEN = (151, 255, 194)
DEAD_LENS = (58, 54, 60)           # an unlit lens in shade: dark, and near enough grey
DEAD_AMBER_GLASS = (0, 187, 220)   # the unlit amber lens in sunlight -- the old trap
HOUSING = (127, 126, 129)          # grey plastic
TAN_WALL = (59, 137, 180)          # the sunlit building behind the light
DARK_FOLIAGE = (50, 105, 60)       # a tree behind the light


def lamp_picture(top, middle, bottom, box_w=40, box_h=60, around=HOUSING, size=(200, 200)):
    """A picture with one lamp head in the middle of it, drawn where the code will look.

    The head box is `box_w` x `box_h`; the lens column is painted into the same slice of it
    that `read_lamp_colour` reads, so this test exercises the real geometry, not a stand-in.
    """
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img[:, :] = (90, 92, 95)                         # dull sky/ground everywhere else
    u, v = size[1] / 2.0, size[0] / 2.0
    left, top_y = u - box_w / 2.0, v - box_h / 2.0
    img[int(top_y):int(top_y + box_h), int(left):int(left + box_w)] = around
    c0 = int(round(left + box_w * LENS_LEFT))
    c1 = int(round(left + box_w * LENS_RIGHT))
    r0 = int(round(top_y))
    r1 = int(round(top_y + box_h * LENS_BOTTOM))
    tall = r1 - r0
    for i, colour in enumerate((top, middle, bottom)):
        a = r0 + (tall * i) // 3
        b = r0 + (tall * (i + 1)) // 3
        img[a:b, c0:c1] = colour
    return img, u, v, float(box_w), float(box_h)


# ---------------------------------------------------------------- it reads the three colours


@pytest.mark.parametrize("lamps,expect", [
    ((LIT_RED, DEAD_LENS, DEAD_LENS), RED),
    ((DEAD_LENS, LIT_AMBER, DEAD_LENS), YELLOW),
    ((DEAD_LENS, DEAD_LENS, LIT_GREEN), GREEN),
])
def test_reads_each_colour(lamps, expect):
    img, u, v, w, h = lamp_picture(*lamps)
    colour, confidence = read_lamp_colour(img, u, v, w, h)
    assert colour == expect
    assert confidence > 0.5


def test_a_dead_light_is_unknown_not_a_guess():
    """Every lens out and in shade. Unknown means stop, so this must not invent a colour."""
    img, u, v, w, h = lamp_picture(DEAD_LENS, DEAD_LENS, DEAD_LENS)
    assert read_lamp_colour(img, u, v, w, h)[0] == UNKNOWN


def test_a_dead_light_in_full_sun_still_means_stop():
    """Honest about a real limit: in sunlight the unlit amber lens is orange glass, so a
    switched-off light reads as YELLOW rather than as nothing. That is a stop either way --
    the rule this module owes the van is only ever that it must not read as GREEN.
    """
    img, u, v, w, h = lamp_picture(DEAD_LENS, DEAD_AMBER_GLASS, DEAD_LENS)
    assert read_lamp_colour(img, u, v, w, h)[0] in (RED, YELLOW, UNKNOWN)


# ---------------------------------------------------------------- the mistakes that matter


def test_the_unlit_amber_lens_does_not_beat_a_lit_green_lamp():
    """The fault that stopped the van at green lights for two attempts.

    In daylight the dead amber glass reads 220/187/0. That is MORE colourful than the lit
    green lamp at 194/255/151, so scoring by colourfulness picked amber every time. The lamp
    that is actually GLOWING is the one whose own channel is nearly full.
    """
    img, u, v, w, h = lamp_picture(DEAD_LENS, DEAD_AMBER_GLASS, LIT_GREEN)
    assert read_lamp_colour(img, u, v, w, h)[0] == GREEN


def test_a_tan_building_filling_the_box_never_says_green():
    """Sunlit tan brick is bright and orange. It may cost us a stop; it may not cost a red."""
    img, u, v, w, h = lamp_picture(TAN_WALL, TAN_WALL, TAN_WALL, around=TAN_WALL)
    assert read_lamp_colour(img, u, v, w, h)[0] != GREEN


def test_a_tree_behind_the_green_lens_does_not_say_green():
    """Dark foliage is green-hued. A lit green lamp measures 160..255 on its own channel and
    foliage sits well below, so the brightness floor tells them apart."""
    assert DARK_FOLIAGE[1] < MIN_LIT_GREEN
    img, u, v, w, h = lamp_picture(DEAD_LENS, DEAD_LENS, DARK_FOLIAGE)
    assert read_lamp_colour(img, u, v, w, h)[0] != GREEN


def test_red_in_the_green_lamps_place_is_not_read_as_red():
    """A colour is only ever reported by its own lamp. Something red low down in the column
    is not a red light -- it is a red thing -- so it must not be called one."""
    img, u, v, w, h = lamp_picture(DEAD_LENS, DEAD_LENS, LIT_RED)
    assert read_lamp_colour(img, u, v, w, h)[0] != RED


def test_a_covered_lens_is_unknown():
    img, u, v, w, h = lamp_picture(HOUSING, HOUSING, HOUSING, around=HOUSING)
    assert read_lamp_colour(img, u, v, w, h)[0] == UNKNOWN


def test_nothing_to_read_is_unknown():
    img, u, v, w, h = lamp_picture(LIT_GREEN, LIT_GREEN, LIT_GREEN)
    assert read_lamp_colour(None, u, v, w, h)[0] == UNKNOWN
    # too far away to resolve the three lamps at all
    assert read_lamp_colour(img, u, v, 1.0, float(MIN_LENS_ROWS))[0] == UNKNOWN
    # off the side of the picture
    assert read_lamp_colour(img, -400.0, v, w, h)[0] == UNKNOWN


# ---------------------------------------------------------------- the map of lamp heads


class FakeBox:
    def __init__(self, x, y, z, ex, ey, ez):
        self.location = type("L", (), {"x": x, "y": y, "z": z})()
        self.extent = type("E", (), {"x": ex, "y": ey, "z": ez})()


class FakeLight:
    def __init__(self, light_id, boxes):
        self.id = light_id
        self._boxes = boxes

    def get_light_boxes(self):
        return self._boxes


class FakeWorld:
    def __init__(self, lights):
        self._lights = lights

    def get_actors(self):
        outer = self

        class Actors:
            def filter(self, _pattern):
                return outer._lights
        return Actors()


def test_lamp_map_keeps_real_heads_and_drops_the_little_repeaters():
    """CARLA reports four boxes per light. Only the three-lamp housings are readable; the
    0.2 m pedestrian repeater near the ground is not, and would only add noise."""
    world = FakeWorld([FakeLight(7, [
        FakeBox(-34.1, -51.0, 4.05, 0.48, 0.20, 0.70),   # the head on the pole
        FakeBox(-43.7, -50.6, 5.15, 0.18, 0.20, 0.59),   # a head out on the mast arm
        FakeBox(-34.4, -50.9, 1.33, 0.03, 0.05, 0.10),   # the little repeater -> dropped
        FakeBox(-34.2, -50.9, 2.91, 0.36, 0.10, 0.25),   # too short to hold three lamps
    ])])
    lamps = LampMap.from_world(world)
    heads = lamps.for_light(7)
    assert len(heads) == 2
    assert all(h.z > 2.0 for h in heads)
    tall = max(heads, key=lambda h: h.height_m)
    assert tall.height_m == pytest.approx(1.40, abs=0.02)


def test_lamp_map_survives_a_light_that_will_not_answer():
    class Broken(FakeLight):
        def get_light_boxes(self):
            raise RuntimeError("no boxes for you")
    world = FakeWorld([Broken(1, []), FakeLight(2, [FakeBox(0, 0, 4.0, 0.4, 0.2, 0.7)])])
    lamps = LampMap.from_world(world)
    assert lamps.for_light(1) == []
    assert len(lamps.for_light(2)) == 1


# ---------------------------------------------------------------- the reader, end to end


def reader_with_head(x=30.0, y=0.0, z=4.05):
    lamps = LampMap({9: [LampHead(light_id=9, x=x, y=y, z=z,
                                  width_m=1.04, height_m=1.40)]})
    return CameraLightReader(lamps)


def picture_of_head(reader, lamps_colours, ego=(0.0, 0.0, 0.0, 0.0)):
    """Draw the lens column exactly where the reader will project it, for this pose."""
    reader.update(np.zeros((600, 800, 3), dtype=np.uint8), ego[0], ego[1], ego[3], ego[2])
    head = reader.lamp_map.for_light(9)[0]
    spot = reader._to_pixels(head)
    assert spot is not None, "the head should be in front of the camera"
    u, v, range_m = spot
    focal = reader.camera.focal_px
    w = head.width_m * focal / range_m
    h = head.height_m * focal / range_m
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:, :] = (90, 92, 95)
    left, top_y = u - w / 2.0, v - h / 2.0
    c0, c1 = int(round(left + w * LENS_LEFT)), int(round(left + w * LENS_RIGHT))
    r0, r1 = int(round(top_y)), int(round(top_y + h * LENS_BOTTOM))
    tall = r1 - r0
    for i, colour in enumerate(lamps_colours):
        a, b = r0 + (tall * i) // 3, r0 + (tall * (i + 1)) // 3
        img[a:b, c0:c1] = colour
    return img


def test_reader_finds_the_head_ahead_and_reads_it():
    reader = reader_with_head()
    img = picture_of_head(reader, (LIT_RED, DEAD_LENS, DEAD_LENS))
    reader.update(img, 0.0, 0.0, 0.0, 0.0, now=1.0)
    assert reader.read(9) == RED


def test_green_has_to_be_seen_twice_but_red_only_once():
    """Green is the only answer that lets the van move, so it needs a second witness. A red
    or an unreadable light is believed at once -- caution never waits."""
    reader = reader_with_head()
    green = picture_of_head(reader, (DEAD_LENS, DEAD_LENS, LIT_GREEN))
    red = picture_of_head(reader, (LIT_RED, DEAD_LENS, DEAD_LENS))

    reader.update(green, 0.0, 0.0, 0.0, 0.0, now=1.0)
    assert reader.read(9) == UNKNOWN          # first sighting: not yet
    reader.update(green, 0.0, 0.0, 0.0, 0.0, now=1.1)
    assert reader.read(9) == GREEN            # confirmed

    reader.update(red, 0.0, 0.0, 0.0, 0.0, now=1.2)
    assert reader.read(9) == RED              # believed immediately


def test_a_stale_green_is_not_a_witness():
    """Two greens a whole second apart are two separate glances, not agreement."""
    reader = reader_with_head()
    green = picture_of_head(reader, (DEAD_LENS, DEAD_LENS, LIT_GREEN))
    reader.update(green, 0.0, 0.0, 0.0, 0.0, now=1.0)
    assert reader.read(9) == UNKNOWN
    reader.update(green, 0.0, 0.0, 0.0, 0.0, now=1.0 + AGREEMENT_MAX_AGE_S + 0.2)
    assert reader.read(9) == UNKNOWN


def test_a_light_behind_us_is_unknown():
    reader = reader_with_head(x=-30.0)
    reader.update(np.zeros((600, 800, 3), dtype=np.uint8), 0.0, 0.0, 0.0, 0.0, now=1.0)
    assert reader.read(9) == UNKNOWN


def test_a_light_we_have_no_head_for_is_unknown():
    reader = reader_with_head()
    reader.update(np.zeros((600, 800, 3), dtype=np.uint8), 0.0, 0.0, 0.0, 0.0, now=1.0)
    assert reader.read(4242) == UNKNOWN


def test_it_drops_straight_into_the_phase_15_lookahead():
    """The whole point of the phase 15 split: the geometry, the lane matching, the distances
    and the rates do not move. Only where the colour comes from changes."""
    reader = reader_with_head()
    img = picture_of_head(reader, (LIT_RED, DEAD_LENS, DEAD_LENS))
    reader.update(img, 0.0, 0.0, 0.0, 0.0, now=1.0)

    signals = SignalMap({9: SignalGeometry(light_id=9, stop_points=[(28.0, 0.0)],
                                           lanes={(7, -1)})})
    look = TrafficLightLookahead(signals, state_source=reader.read)
    assert look.state_source(9) == RED
