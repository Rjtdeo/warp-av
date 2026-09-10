"""Read the traffic light's colour off the camera, instead of asking the simulator.

Perception V2, phase 15b.

Phase 15 gave the van a map of every traffic light: where each one stands, which lanes it
governs, and how far ahead the next one is. It still asked CARLA what colour the light was,
which is not a question a real van can ask anybody. This module answers it from pixels.

It does NOT go hunting for the light in the picture. The map already says where the lamp head
is, to the centimetre, and a real van's HD map records the same thing. So the work splits:

    the map says WHERE to look       -- geometry, never changes, built once
    the camera says WHAT COLOUR      -- pixels, the only thing read live

Everything below was measured on Town10HD, and three of the measurements killed a rule that
had looked obvious:

1. CARLA's lamp-head box is far bigger than the lamps. The box is 1.4 m tall and about a
   metre across where the lens column is roughly 0.5 m by 0.75 m; the rest is the bracket
   back to the pole, and sky and building past the edges. Reading the whole box meant reading
   the BUILDING BEHIND THE LIGHT, and sunlit tan brick is bright, orange, and reads as yellow.

   Finding the lens column inside the box took two goes, and the first was wrong in a way
   worth recording. Flipping every light through all three colours and keeping the pixels
   that CHANGED is the right method -- but the van was placed by walking back down the lane a
   metre at a time, and that walk BRANCHES at junctions. Several lights were therefore
   measured from a different street, side-on. The answer that came out (lens at 48%..76%
   across, top 62% down) fitted those views and was wrong for a straight approach: on light
   837 the strip it picked was pure tan building, identical on red and on green.

   Measured again from straight approaches only -- refusing any branch that turns more than
   twelve degrees -- the lens column is CENTRED in the box, spanning 30%..78% across and
   18%..82% down. That is what is used here.

2. "The lit lamp is the most colourful thing in the box" is wrong. The UNLIT amber lens is
   orange glass: in daylight it reads 220/187/0, MORE colourful than a lit green lamp at
   183/254/140, which carries a lot of white. Judging by colourfulness picks the wrong lamp.

3. What really separates a lit lamp from a dead one is that it GLOWS -- its own colour channel
   runs close to full. Scoring each lamp by the brightness of its own channel, rather than by
   how colourful it is, is what made the reading work at all.

Measured over 300 readings on straight approaches: fourteen lights, eight distances from 92 m
in to 13 m, all three colours at each. What matters is not the colour name but what the van
DOES with it -- go on green, stop on anything else:

    goes when the light really is green      88 of 100
        under 25 m                           100%
        25 to 45 m                            97%
        over 45 m                             80%
    goes when the light is NOT green          0 of 200
    stops for every red                     100 of 100

Not one false go. Two things hold that line. A colour has to come from the right part of the
column -- green only from the lower two thirds, red only from the upper three quarters -- so
a green tree above the light or a red sign below it cannot say go. And green, the only answer
that lets the van move, must be seen twice running before it is believed; red, yellow and
"cannot read it" are believed the first time.

The honest limit: beyond 45 m the lamp is two or three pixels across, and one green in five is
read as amber. That costs a slow-down, never a red light run, and by 25 m every green in the
set was read correctly.

Swapping this in is one argument, exactly as phase 15 left room for:

    reader = CameraLightReader(LampMap.from_world(world))
    lookahead = TrafficLightLookahead(signal_map, state_source=reader.read)

...with reader.update(front_camera_image, ego_x, ego_y, ego_yaw_deg, ego_z) once a tick.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .camera_model import VIEW_MOUNTS, CameraModel
from .traffic_lights import GREEN, RED, UNKNOWN, YELLOW

# ---------------------------------------------------------------------------------------
# Where the lamps sit inside CARLA's lamp-head box.
# Not guessed: every light was flipped through all three colours from a STRAIGHT approach and
# the pixels that changed were recorded. The straightness matters -- see the module docstring
# for the first attempt, which measured several lights side-on and got a different answer.
# ---------------------------------------------------------------------------------------
LENS_LEFT = 0.30            #: the lens column starts this far across CARLA's box
LENS_RIGHT = 0.78           #: ... and ends here. The rest is bracket, sky and building.
LENS_TOP = 0.18             #: the lamps start this far down the box
LENS_BOTTOM = 0.82          #: ... and end here.

#: Which part of the column each colour may come from. A traffic light is stacked red / amber
#: / green from the top, so a green thing at the TOP of the column is not a green lamp and a
#: red thing at the BOTTOM is not a red lamp. These two lines are deliberately loose and
#: overlapping: a hard split into exact thirds was tried and cost fourteen points, because
#: CARLA's box does not line up with the lamps closely enough to trust thirds. Loose as they
#: are they still cost nothing -- 88 correct greens out of 100 with them, 87 without -- and
#: they are what stops a tree above the light from ever saying go.
GREEN_ONLY_BELOW = 0.34     #: green may only come from the lower two thirds
RED_ONLY_ABOVE = 0.75       #: red may only come from the upper three quarters

MIN_LENS_ROWS = 3           #: under one row per lamp there is nothing to read
MIN_CHROMA = 20.0           #: greyer than this is housing, not a lens

# Hue boundaries, degrees. A lit red lamp measures 8-27, amber 44-58, green 87-131, so these
# sit in the empty gaps between them rather than on the edge of anything.
RED_HUE_MAX = 34.0
YELLOW_HUE_MAX = 68.0
GREEN_HUE_MAX = 168.0

#: A lit green lamp measures 160..255 in its green channel, even at 92 m. Dark green things
#: that are NOT lamps -- foliage, a painted sign in shadow -- sit well below that. Without
#: this floor a tree standing behind the green lens could hand the van a green light, which
#: is the one mistake this module must never make. Measured: 130 costs nothing at any range.
MIN_LIT_GREEN = 130.0

GREEN_NEEDS_TWO = True      #: green is the only answer that lets the van go, so ask twice
AGREEMENT_MAX_AGE_S = 1.0   #: ... but a read from a second ago is not a witness any more
FULL_CONFIDENCE_LIT = 220.0 #: a lamp channel this bright is as sure as it gets


def camera_lights_wanted(env=None) -> bool:
    """WARP_CAMERA_LIGHTS=0 goes back to asking the simulator, for a side-by-side run."""
    env = os.environ if env is None else env
    return str(env.get("WARP_CAMERA_LIGHTS", "1")).strip().lower() not in ("0", "false", "off", "no")


# ---------------------------------------------------------------------------------------
# A. WHERE the lamps are -- read once, like the signal map
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LampHead:
    """One stack of three lenses, in world coordinates. Never moves."""

    light_id: int
    x: float
    y: float
    z: float
    width_m: float          #: how wide CARLA's box is, across the van's line of sight
    height_m: float         #: how tall CARLA's box is


class LampMap:
    """Every traffic light's lamp heads. Built ONCE, next to the SignalMap."""

    #: a head is a real three-lamp housing; the little pedestrian repeaters are far smaller
    MIN_HEAD_Z_M = 2.0
    MIN_HEAD_HALF_HEIGHT_M = 0.4

    def __init__(self, heads: Optional[Dict[int, List[LampHead]]] = None):
        self.heads: Dict[int, List[LampHead]] = dict(heads or {})
        self.build_ms = 0.0

    def __len__(self) -> int:
        return len(self.heads)

    def for_light(self, light_id: int) -> List[LampHead]:
        return self.heads.get(int(light_id), [])

    @classmethod
    def from_world(cls, world) -> "LampMap":
        """Ask CARLA where every lamp head is. Geometry only -- no colours are read here."""
        t0 = time.perf_counter()
        found: Dict[int, List[LampHead]] = {}
        try:
            lights = list(world.get_actors().filter("traffic.traffic_light*"))
        except Exception:
            lights = []
        for tl in lights:
            try:
                boxes = tl.get_light_boxes()
            except Exception:
                continue
            heads: List[LampHead] = []
            for box in boxes:
                loc, ext = box.location, box.extent
                if float(loc.z) < cls.MIN_HEAD_Z_M:
                    continue
                if float(ext.z) < cls.MIN_HEAD_HALF_HEIGHT_M:
                    continue
                heads.append(LampHead(
                    light_id=int(tl.id),
                    x=float(loc.x), y=float(loc.y), z=float(loc.z),
                    # the box is axis-aligned and we may approach it from any side, so take
                    # the width across the diagonal: never narrower than what we will see
                    width_m=2.0 * math.hypot(float(ext.x), float(ext.y)),
                    height_m=2.0 * float(ext.z),
                ))
            if heads:
                found[int(tl.id)] = heads
        out = cls(found)
        out.build_ms = (time.perf_counter() - t0) * 1000.0
        return out


# ---------------------------------------------------------------------------------------
# B. WHAT COLOUR -- the only thing read from pixels
# ---------------------------------------------------------------------------------------


def _hue_chroma(patch: np.ndarray):
    """Hue in degrees and colourfulness, for a BGR patch. No OpenCV: this runs every tick."""
    px = patch.astype(np.float32)
    b, g, r = px[..., 0], px[..., 1], px[..., 2]
    top = px.max(axis=-1)
    chroma = top - px.min(axis=-1)
    safe = np.where(chroma == 0.0, 1.0, chroma)
    hue = np.where(top == r, 60.0 * (((g - b) / safe) % 6.0),
                   np.where(top == g, 60.0 * ((b - r) / safe + 2.0),
                            60.0 * ((r - g) / safe + 4.0)))
    return hue, chroma, r, g


def read_lamp_colour(image: np.ndarray, u: float, v: float,
                     width_px: float, height_px: float) -> Tuple[str, float]:
    """The colour of the lamp whose head box lands at (u, v) with this size, from pixels.

    Each colour is scored by the BRIGHTEST PIXEL OF ITS OWN CHANNEL that also carries its own
    hue. A lit lamp is one that GLOWS, so its own channel runs close to full. Colourfulness is
    deliberately not the test: in daylight the unlit amber lens is orange glass reading
    220/187/0, which beats a lit green lamp at 183/254/140, and scoring that way stopped the
    van at green lights.

    On top of that, green may only be found in the lower part of the column and red only in
    the upper part, because that is how a traffic light is stacked.
    """
    if image is None or getattr(image, "size", 0) == 0:
        return UNKNOWN, 0.0
    rows, cols = image.shape[0], image.shape[1]
    left_x = u - width_px / 2.0
    top_y = v - height_px / 2.0
    u0 = max(0, int(round(left_x + width_px * LENS_LEFT)))
    u1 = min(cols, int(round(left_x + width_px * LENS_RIGHT)))
    v0 = max(0, int(round(top_y + height_px * LENS_TOP)))
    v1 = min(rows, int(round(top_y + height_px * LENS_BOTTOM)))
    if u1 - u0 < 1 or v1 - v0 < MIN_LENS_ROWS:
        return UNKNOWN, 0.0

    strip = image[v0:v1, u0:u1]
    tall = strip.shape[0]
    hue, chroma, red_ch, green_ch = _hue_chroma(strip)
    green_starts = int(tall * GREEN_ONLY_BELOW)
    red_ends = max(1, int(tall * RED_ONLY_ABOVE))

    scores = {}
    for colour in (RED, YELLOW, GREEN):
        if colour == RED:
            h, c, lit = hue[:red_ends], chroma[:red_ends], red_ch[:red_ends]
            # red wraps past zero, so it is "below 34 degrees or above 340"
            matches = (h < RED_HUE_MAX) | (h >= 340.0)
        elif colour == YELLOW:
            h, c = hue, chroma
            lit = np.minimum(red_ch, green_ch)      # amber needs BOTH channels up
            matches = (h >= RED_HUE_MAX) & (h < YELLOW_HUE_MAX)
        else:
            h, c = hue[green_starts:], chroma[green_starts:]
            lit = green_ch[green_starts:]
            matches = ((h >= YELLOW_HUE_MAX) & (h < GREEN_HUE_MAX)
                       & (lit >= MIN_LIT_GREEN))
        if c.size == 0:
            scores[colour] = 0.0
            continue
        matches = matches & (c >= MIN_CHROMA)
        scores[colour] = float(lit[matches].max()) if matches.any() else 0.0

    best, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score <= 0.0:
        return UNKNOWN, 0.0
    return best, min(1.0, best_score / FULL_CONFIDENCE_LIT)


# ---------------------------------------------------------------------------------------


@dataclass
class _Recent:
    colour: str = UNKNOWN
    when: float = 0.0


class CameraLightReader:
    """Answers "what colour is light N" from the front camera, for the phase 15 lookahead.

    Feed it a frame and the van's pose once a tick; hand `read` to TrafficLightLookahead.
    """

    def __init__(self, lamp_map: LampMap, camera: Optional[CameraModel] = None):
        self.lamp_map = lamp_map
        if camera is None:
            mount, pitch, yaw, w, h, fov = VIEW_MOUNTS["front"]
            camera = CameraModel(width=w, height=h, fov_deg=fov, mount=mount,
                                 pitch_deg=pitch, yaw_deg=yaw, name="front")
        self.camera = camera
        self._image: Optional[np.ndarray] = None
        self._pose = (0.0, 0.0, 0.0, 0.0)     # x, y, z, yaw degrees
        self._frame_time = 0.0
        self._recent: Dict[int, _Recent] = {}
        self.last_confidence = 0.0
        self.last_raw = UNKNOWN
        self.reads = 0
        self.read_ms = 0.0

    # -- once a tick -------------------------------------------------------------------
    def update(self, image, ego_x: float, ego_y: float, ego_yaw_deg: float,
               ego_z: float = 0.0, now: Optional[float] = None) -> None:
        """The latest front-camera picture and where the van was when it was taken."""
        self._image = image
        self._pose = (float(ego_x), float(ego_y), float(ego_z), float(ego_yaw_deg))
        self._frame_time = time.time() if now is None else float(now)

    # -- the geometry ------------------------------------------------------------------
    def _to_pixels(self, head: LampHead) -> Optional[Tuple[float, float, float]]:
        """Where this lamp head lands in the picture, and how far away it is."""
        ex, ey, ez, eyaw = self._pose
        a = math.radians(eyaw)
        ca, sa = math.cos(a), math.sin(a)
        dx, dy = head.x - ex, head.y - ey
        # world -> the van's own frame, then -> the LiDAR frame the camera model speaks
        fx = dx * ca + dy * sa
        fy = -dx * sa + dy * ca
        fz = head.z - ez
        lx = fx - self.camera.lidar_mount[0]
        ly = fy - self.camera.lidar_mount[1]
        lz = fz - self.camera.lidar_mount[2]
        uv = self.camera.project(lx, ly, lz)
        if uv is None:
            return None
        ahead = fx - self.camera.mount[0]
        if ahead <= 0.5:
            return None
        return uv[0], uv[1], ahead

    # -- the answer --------------------------------------------------------------------
    def read(self, light_id: int) -> str:
        """light id -> "red" | "yellow" | "green" | "unknown". The phase 15 state source."""
        started = time.perf_counter()
        try:
            colour, conf = self._read_now(int(light_id))
        finally:
            self.read_ms = (time.perf_counter() - started) * 1000.0
            self.reads += 1
        self.last_raw = colour
        self.last_confidence = conf

        if not GREEN_NEEDS_TWO or colour != GREEN:
            self._remember(light_id, colour)
            return colour
        # Green is the only answer that lets the van move, so it has to be said twice.
        seen = self._recent.get(int(light_id))
        agreed = (seen is not None and seen.colour == GREEN
                  and (self._frame_time - seen.when) <= AGREEMENT_MAX_AGE_S)
        self._remember(light_id, GREEN)
        return GREEN if agreed else UNKNOWN

    def _remember(self, light_id: int, colour: str) -> None:
        self._recent[int(light_id)] = _Recent(colour=colour, when=self._frame_time)

    def _read_now(self, light_id: int) -> Tuple[str, float]:
        if self._image is None:
            return UNKNOWN, 0.0
        heads = self.lamp_map.for_light(light_id)
        if not heads:
            return UNKNOWN, 0.0
        rows, cols = self._image.shape[0], self._image.shape[1]
        camera = self.camera.with_frame(cols, rows, self.camera.fov_deg)
        if camera is not self.camera:
            self.camera = camera
        best: Tuple[str, float] = (UNKNOWN, 0.0)
        best_size = 0.0
        for head in heads:
            spot = self._to_pixels(head)
            if spot is None:
                continue
            u, v, range_m = spot
            focal = self.camera.focal_px
            width_px = head.width_m * focal / range_m
            height_px = head.height_m * focal / range_m
            if u < -width_px or u > cols + width_px or v < -height_px or v > rows + height_px:
                continue
            colour, conf = read_lamp_colour(self._image, u, v, width_px, height_px)
            # more than one head can show the same light; trust the one we see best, and
            # never let a blank one erase a good read
            if colour != UNKNOWN and height_px > best_size:
                best, best_size = (colour, conf), height_px
        return best
