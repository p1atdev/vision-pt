#!/bin/bash

source .venv/bin/activate

accelerate launch train/jit/class_to_image_tread.py $@
