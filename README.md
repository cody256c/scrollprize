# scrollprize

Vesuvius Challenge

## v07

Prepare a virtual enviornment:

```
DEST="tmp/venv"

mkdir $DEST -p

python3 -m venv $DEST

source $DEST/bin/activate

python3 -m pip install --no-cache-dir --upgrade pip

python3 -m pip install --no-cache-dir albumentations
python3 -m pip install --no-cache-dir lightning
python3 -m pip install --no-cache-dir matplotlib
python3 -m pip install --no-cache-dir numpy
python3 -m pip install --no-cache-dir opencv-python
python3 -m pip install --no-cache-dir pandas
python3 -m pip install --no-cache-dir Pillow
python3 -m pip install --no-cache-dir segmentation_models_pytorch
python3 -m pip install --no-cache-dir torch torchvision torchaudio
python3 -m pip install --no-cache-dir torch-summary
python3 -m pip install --no-cache-dir tqdm

deactivate
```

Run training and inference scripts:

```
# activate venv
source tmp/venv/bin/activate

# navigate to src/v07
cd src/v07

# train
# python3 main.py train --n_cfg=0
python3 main.py train --cfg=cfg/cfg_12.json

# inference
# python3 main.py infer --n_cfg=0 --segment_id=20231005123336
python3 main.py infer --cfg=cfg/cfg_12.json --segment_id=20231005123336
```
