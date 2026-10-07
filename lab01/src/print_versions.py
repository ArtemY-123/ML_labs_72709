import sys
import codecarbon
import fastapi
import httpx
import joblib
import locust
import matplotlib
import memory_profiler
import mlflow
import numpy as np
import onnx
import onnxruntime
import pandas as pd
import psutil
import pyarrow
import pytest
import requests
import scipy
import seaborn as sns
import sklearn
import thop
import torch
import torchinfo
import torchvision
import tqdm
import uvicorn

packages = {
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit-learn": sklearn.__version__,
    "scipy": scipy.__version__,
    "matplotlib": matplotlib.__version__,
    "seaborn": sns.__version__,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torchinfo": torchinfo.__version__,
    "thop": thop.__version__,
    "onnx": onnx.__version__,
    "onnxruntime": onnxruntime.__version__,
    "mlflow": mlflow.__version__,
    "memory-profiler": memory_profiler.__version__,
    "psutil": psutil.__version__,
    "codecarbon": codecarbon.__version__,
    "fastapi": fastapi.__version__,
    "uvicorn": uvicorn.__version__,
    "pytest": pytest.__version__,
    "httpx": httpx.__version__,
    "locust": locust.__version__,
    "requests": requests.__version__,
    "pyarrow": pyarrow.__version__,
    "joblib": joblib.__version__,
    "tqdm": tqdm.__version__,
}

for name, ver in packages.items():
    print(f"{name}=={ver}")