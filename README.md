# scrollprize
Vesuvius Challenge

```
# train
python3 main.py train --labels=../../input/set01 --out=../../output/v03 --data=../../dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths
#infer
python3 main.py infer --segment_id=20230925002745 --out=../../output/v03/prediction --data=../../dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths --model=../../output/v03/lightning_logs/version_0/checkpoints/last.ckpt
```
