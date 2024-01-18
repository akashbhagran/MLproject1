#!/bin/sh

for f in model/*.py; do python -m black "$f"; done

for f in unittests/*.py; do python -m black "$f"; done
 