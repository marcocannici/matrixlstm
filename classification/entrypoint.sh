#!/bin/bash
cd /exp/layers/extensions && \
python setup.py install --user && \
cd /exp && \
/bin/sh -c "$*"
