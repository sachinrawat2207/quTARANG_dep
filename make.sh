#!/bin/sh
python3 -m build
python3 -m pip install dist/gpe-0.0.1-py3-none-any.whl --force-reinstall