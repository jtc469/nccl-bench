#!/bin/bash

python3 analysis/summarise.py scc \
    --labels baseline graphs graphs-4g graphs-ll graphs-ch2 \
    --exclude 465 466 467 468 469