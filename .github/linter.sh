#!/bin/bash

cd ../model

python -m black data.py
python -m black train.py