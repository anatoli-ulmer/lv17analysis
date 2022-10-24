#!/bin/bash

set -eux

# only test codestyle
python -m pycodestyle lv17analysis h5analysis
