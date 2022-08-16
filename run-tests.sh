#!/bin/bash

set -eux

# only test codestyle
python -m pycodestyle lv17analysis --statistics --count --show-source --ignore=W391,E123,E226,E24,W504 --max-line-length=99
