# Blur a portrait, with optional framing

Machine: M1 Pro, Fedora 44 aarch64. M2, M4 Pro and Neo use CPU until their own
model packages and generation owners are qualified. No camera or audio opens.

After the lead's install step in [LEAD-RUN.md](LEAD-RUN.md), run from this directory:

```sh
.venv-h13/bin/python try-camera-h13 preview --input /path/to/portrait.jpg --engine auto --framing
```

Expect startup to print `engine=ane-hybrid` for a qualified H13 package, or
`engine=cpu` with a reason. Open the printed PNG path to see blur and framing.
Use `--engine cpu` to compare. `--engine ane` requests ANE and retains automatic
CPU fallback if unavailable or faulty. The still preview stretches model input
as in the existing backend; this is not a temporal camera qualification.

Off:

```sh
.venv-h13/bin/python try-camera-h13 off
```

Off removes the preview cache. The process has already exited; no service,
default camera, microphone, PipeWire route or system setting was changed.
The backend can be used by the PR #3 camera consumer after qualification.

To make an openly licensed test portrait available without a personal image:

```sh
.venv-h13/bin/python h13/images.py /tmp/camera-h13-portraits
```

Then use `/tmp/camera-h13-portraits/000000000872.jpg` as the input. The image is
COCO validation image 872, CC BY 2.0; retain its COCO/Flickr metadata attribution.
