# PhantomBrush

A real-time hand-tracking drawing app. Paint in the air using your camera and your fingertip.

## How It Works
PhantomBrush uses computer vision to track your hand in real time via web cam. 
Raise your index finger to draw; raise your index and middle fingers together to lift the "pen" and move without drawing. Strokes render onto a persistent canvas layered over the live camera feed.

## Tech Stack
- Python
- OpenCV (video capture, drawing, frame processing)
- MediaPipe Tasks API ('HandLandmarker') for real-time hand landmark detection

## Features
- Real-time hand and fingertip tracking via webcam
- Draw by moving your index finger
- Pen-up / pen-down gesture control (index-only draws, index+middle lifts the pen)
- Jump-distance filtering to prevent stray lines from connecting across tracking gaps
- Clear canvas on demand (`c` key)
- [Planned] Gesture-based color picker
- [Planned] Save drawings as PNGs
- [Planned] Multi-hand support

## Installation
git clone https://github.com/AstronautGib/phantom-brush.git
cd phantom-brush
python -m venv venv
venv\Scripts\activate #Windows
pip install -r requirements.txt

(Download the latest model of mediapipe and place it in models folder: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)

python main.py

## Usage
- Raise only your index finger to draw
- Raise index + middle fingers together (peace sign) to move without drawing
- Press `c` to clear the canvas
- Press `q` or close the window to quit

## What I Learned
While building this, MediaPipe restructured its Python API and removed the legacy `mp.solutions` interface I originally built against. Rather than pin an older, deprecated version, I migrated the hand-tracking layer to MediaPipe's newer Tasks API (`HandLandmarker`), which meant reworking how landmark results are read and how the model itself is loaded (as an explicit `.task` file instead of a bundled solution). It was a good lesson in handling breaking changes in a fast-moving CV library and keeping a project on a current, actively-maintained API rather than a soon-to-be-deprecated one.

I also ran into environment-specific issues worth noting for future projects: WSL doesn't have native access to host webcams, and venvs created on one machine don't reliably transfer to another (hardcoded paths break when usernames differ). Both are now avoided by rebuilding the venv locally per machine.

## Future Improvements
- [ ] Gesture-based color picker
- [ ] Save drawings as images
- [ ] Multi-hand support
- [ ] Possible spin-off: a lightweight game (e.g. shape-tracing challenge) built on the same hand-tracking mechanic

## License
MIT