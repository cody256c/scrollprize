#!/bin/bash

# nohup ./infer.sh 12 > logging.txt &

. ../../tmp/venv/bin/activate

segments=(
"20230702185753"
"20230827161847"
"20230925002745"
"20230929220926"
"20231005123336"
"20231007101616"
"20231012173610"
"20231012184421_superseded"
"20231012184423"
"20231016151002"
"20231022170900"
"20231022170901"
"20231031143852"
"20231106155351"
"20231210121321"
"20231221180250"
)

for seg in ${segments[@]}; do
	echo $seg
	python3 main.py infer --cfg=cfg/cfg_$1.json --segment_id=$seg
done
