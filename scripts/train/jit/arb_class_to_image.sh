#!/bin/bash

source .venv/bin/activate

accelerate launch train/jit/arb_class_to_image.py $@
