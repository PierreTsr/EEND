# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.

import sys
import chainer
try:    
    import cupy
    import cupy.cuda
    from cupy.cuda import cudnn
except:
    print("Warning, you are running this code without CuPy or CUDA. This is highly untested.")


def print_system_info():
    pyver = sys.version.replace('\n', ' ')
    print(f"python version: {pyver}")
    print(f"chainer version: {chainer.__version__}")
    try:
        print(f"cupy version: {cupy.__version__}")
        print(f"cuda version: {cupy.cuda.runtime.runtimeGetVersion()}")
        print(f"cudnn version: {cudnn.getVersion()}")
    except:
        print("No CUDA backend.")
