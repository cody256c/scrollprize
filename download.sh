#!/bin/bash
#export PATH=$PATH:/scratch/$USER/bin/rclone-v1.64.2-linux-amd64

rclone copy :http:/ ./dl.ash2txt.org/ \
    --http-url http://registeredusers:only@dl.ash2txt.org/ \
    --progress \
    --multi-thread-streams=4 \
    --transfers=4 \
    --size-only \
    --filter-from filter-file.txt
