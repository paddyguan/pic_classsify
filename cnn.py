# -*- coding: utf-8 -*-

from __future__ import print_function
from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range

#将所有的图片resize成100*100
w=100
h=100
c=3