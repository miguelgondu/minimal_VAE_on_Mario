#!/bin/sh
Xvfb :99 -screen 0 1366x768x24 -ac +extension GLX +render -noreset &
python3 ./simulator.py