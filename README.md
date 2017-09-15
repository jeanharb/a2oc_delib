# When Waiting is not an Option : Learning Options with a Deliberation Cost

Arxiv link: https://arxiv.org/pdf/1709.04571.pdf

## Installation

Here's a list of all dependencies:

- Numpy
- Theano
- Lasagne
- Argparse
- OpenAI Gym [Atari]
- matplotlib
- cv2 (OpenCV)
- PIL (Image)

## Training

To train, run following command:
```
python train.py --sub-env Breakout --num-options 8 --num-threads 16 --folder-name Breakout_model
```

To view a list of available parameters, run:
```
print train.py --help
```

During training, you can run utils/plot.py to view the training curve. Every argument given can be a path to a different run, which will put all runs on the same plot.
```
python utils/plot.py models/Breakout_model/ models/Breakout_model_v2/ models/Breakout_model_v3/
```

## Testing

To watch model after training, run watch.py and give it the path the saved model files. e.g.:
```
python watch.py models/Breakout_model/
```

