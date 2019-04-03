#!/bin/bash

python3 setup.py clean --all
rm -rf build
rm -rf dist
rm -rf src/ykfan_utils.egg-info
# python3 -m pip install --user --upgrade setuptools wheel
python3 setup.py sdist bdist_wheel
# python3 -m pip install --user --upgrade twine
python3 -m twine upload -u ykfan87 dist/*