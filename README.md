# Evergreen

A rotating Christmas tree built entirely from code. Instead of using a tree image or a 3D model, Evergreen builds its branches, needles, ornaments, and lights from generated points in 3D space, then renders them as tiny ASCII-style characters.

The animation starts with the tree growing from the ground up. Lights spiral around the branches, ornaments appear, presents settle beneath the tree, and a glowing star finishes the scene. The camera keeps moving as the completed tree turns against a snowy night sky.

## Features

- Procedural 3D spruce with uneven boughs, secondary branches, and hanging tips
- Bitmap ASCII-style needles rendered with depth sorting
- Rotating camera and a five-turn light garland wrapped around the tree
- Colored bulbs that cast light onto nearby foliage
- Animated ornaments, presents, snowfall, and an eight-point star
- Source-code panel that scrolls through the actual Python file
- Live desktop playback, still-image capture, and MP4 export

## Getting started

You'll need Python 3.10 or newer.

Install the required packages:

```bash
python -m pip install numpy opencv-python pillow
```

For faster nearest-light calculations, you can also install SciPy:

```bash
python -m pip install scipy
```

Tkinter is used for the live window. It's included with many Python installations; on some Linux systems, you'll need to install `python3-tk` separately.

Run the animation:

```bash
python evergreen.py
```

## Controls

| Key | Action |
|---|---|
| `Space` | Pause / resume |
| `R` | Restart |
| `S` | Toggle rotation |
| `L` | Toggle lights |
| `N` | Toggle snow |
| `E` | Show / hide source-code panel |
| `F` | Toggle fullscreen |
| `Esc` or `Q` | Quit |

## Exporting

Save a still image at a specific point in the animation:

```bash
python evergreen.py --still 14
```

This writes `evergreen_preview.png` to the current directory.

Export the full 20-second sequence:

```bash
python evergreen.py --export evergreen.mp4 --seconds 20 --fps 24
```

For a 1080 × 1920 portrait video, install FFmpeg and use:

```bash
python evergreen.py --export evergreen.mp4 --seconds 20 --fps 24 --scale 1.8
```

You can leave out individual elements when exporting:

```bash
python evergreen.py --export clean.mp4 --no-editor --no-snow
```

Without FFmpeg, the script can still export an MP4 at its native 600 × 1066 resolution. Higher-resolution scaling requires FFmpeg.

## How it works

`build_geometry()` constructs the tree once, using seeded randomness to shape staggered branch tiers and individual needle clusters. A simple 3D camera projection rotates those points around the vertical axis. During rendering, a depth-aware grid keeps the nearest character in each screen cell, which helps the foliage stay readable instead of becoming a blur of overlapping symbols.

The garland, bulbs, ornaments, and presents are stored as 3D coordinates too, so they remain attached to the tree as it rotates. Nearby foliage picks up the bulb colors, while a two-stage blur adds a softer glow around the brighter parts of the scene.

Everything is rendered procedurally with NumPy and OpenCV. Pillow handles the live Tkinter display and image output; SciPy can speed up the light-to-needle lookup. No external textures or 3D assets are needed.

## Notes

- The animation loops every 20 seconds in live mode.
- The default output is 600 × 1066 pixels at a target of 24 FPS. Live performance depends on your computer.
- The MP4 exporter creates a silent video; add audio separately if needed.
- To adjust the look, start with `build_geometry()`, `COLORS`, `EvergreenRenderer.angle()`, and the animation timings inside the drawing methods.

---

A small creative-coding experiment in geometry, typography, and light.
