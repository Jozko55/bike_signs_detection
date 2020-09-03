# bike_signs_detection

## Installation

1. Create a conda enviroment with python 3.6.5 and activate it. \
`conda create --name sign_det python=3.6.5` \
`conda activate sign_det`

1. Install tensorflow 1.13.1 and tensorflow-gpu 1.13.1. \
`pip install tensorflow==1.13.1` \
`pip install tensorflow-gpu==1.13.1`

1. Install keras-retinanet 0.5.1:
   * Clone [keras-retinanet](https://github.com/fizyr/keras-retinanet) repository.  \
   `git clone https://github.com/fizyr/keras-retinanet.git`

   * Move to the keras-retinanet folder. \
   `cd keras-retinanet`

   * Create a branch from 0.5.1 and switch to it. \
   `git checkout -b branch0.5.1 0.5.1`

   * Install it. \
   `pip install .`

1. Install keras-maskrcnn 0.2.2:
   * Clone [keras-maskrcc](https://github.com/fizyr/keras-maskrcnn) repository. \
   `git clone https://github.com/fizyr/keras-maskrcnn.git`

   * Move to the keras-maskrcnn folder. \
   `cd keras-maskrcnn`

   * Create a branch from 0.2.2 and switch to it. \
   `git checkout -b branch0.2.2 0.2.2`

   * Install it. \
   `pip install .`

1. Install keras 2.2.5 \
`pip install keras==2.2.5`



