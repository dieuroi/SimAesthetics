# SimAesthetics

This is a simple ongoing image aesthetic assessment task. 
 
Thanks to the coworker: Kun Xiong. We also thank all the advisers!

## Installing

```bash
git clone https://github.com/dieuroi/SimAesthetics.git
cd SimAesthetics
virtualenv -p python3.8 env
source ./env/bin/activate
```


## Dataset

The model was trained on the [AVA (Aesthetic Visual Analysis) dataset](http://refbase.cvc.uab.es/files/MMP2012a.pdf).
You can get it from [here](https://github.com/mtobeiyf/ava_downloader).
The meta data can be found from [here](https://github.com/dieuroi/AVA_review_downloader).

## Models for this task
- [NIMA](https://arxiv.org/abs/1709.05424): NIMA: Neural Image Assessment.
- [MLSP](https://ieeexplore.ieee.org/document/8953497): Effective Aesthetics Prediction with Multi-Level Spatially Pooled Features.

## Pre-trained model (In Progress)

```bash

```


## Usage
```bash

Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  get_image_score  Get image scores
  prepare_dataset  Parse, clean and split dataset
  train_model      Train model
  validate_model   Validate model
```


## Contributing

Contributing are welcome


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Most of this code is borrowed heavily from the Pytorch NIMA [PyTorch NIMA: Neural IMage Assessment](https://github.com/truskovskiyk/nima.pytorch)
* [pytorch-mobilenet-v2](https://github.com/tonylins/pytorch-mobilenet-v2)
* [origin NIMA article](https://arxiv.org/abs/1709.05424)
* [origin MobileNetV2 article](https://arxiv.org/pdf/1801.04381)
* [Post at Google Research Blog](https://research.googleblog.com/2017/12/introducing-nima-neural-image-assessment.html)
* [origin MLSP article](https://ieeexplore.ieee.org/document/8953497)
* [MLSP (tensorflow)](https://github.com/subpic/ava-mlsp)