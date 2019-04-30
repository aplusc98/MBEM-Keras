# MBEM-Keras
This repository gives a Keras implementation of the MBEM algorithm proposed in the paper Learning From Noisy Singly-labeled Data published at ICLR 2018. The original implementation in MXNet is given at https://github.com/khetan2/MBEM 



USAGE INSTRUCTIONS:

Update the path in the data_loader.py

python3 MBEM-keras.py

ISSUES:
Entire code is compatible with python 2.X but while running on harware accelerated systems running with python 2 does not use GPU.

REQUIREMENTS:

Python 3.X
Keras
Tensorflow (backend)
