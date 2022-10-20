#!/bin/bash

set -eux

# Run autopep8 for automatic codestyle formating. For parameter and exclusions see '.pep8'
autopep8 lv17analysis/*.py h5analysis/*.py

# Check codestyle after formatting
python -m pycodestyle lv17analysis h5analysis
