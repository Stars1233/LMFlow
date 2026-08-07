#!/bin/bash

python -m pytest -q --strict-markers \
  -m "not gpu and not slow and not online and not optional_backend"
