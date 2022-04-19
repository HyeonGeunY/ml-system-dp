import argparse
import json
import logging
import os
import time
from typing import Dict, List, Tuple, Union


import grpc
import mlflow
import numpy as np
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from src.proto import onnx_ml_pb2, predict_pb2, prediction_service_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PytorchImagePreprocessTransformer(
    BaseEstimator,
    TransformerMixin,
):
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (32, 32),
        prediction_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
        mean_vec: List[float] = [0.485, 0.456, 0.406],
        stddev_vec: List[float] = [0.229, 0.224, 0.225],
    ):
        
        self.image_size = image_size
        self.prediction_shape = prediction_shape
        self.mean_vec = mean_vec
        self.stddev_vec = stddev_vec
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: Union[Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(X, np.ndarray):
            # 인풋의 타입, 크기 확인
            dim_0 = (3,) + self.image_size
            dim_1 = self.image_size + (3,)
            if X.shape != dim_0 and X.shape != dim_1:
                raise ValueError(f"resize to image_size {self.image_size} beforehand for numpy array")
            
            # X를 (width, height, channel)로 바꾼다.
            if X.shape == dim_0:
                X = X.transpose(1, 2, 0)
        else:
            # 인풋이 ndarray가 아닐 경우(Image.Image) 이미지 크기맞추고, 형 변환
            X = np.array(X.resize(self.image_size))
        
        image_data = X.transpose(2, 0, 1).astype(np.float32)
        mean_vec = np.array(self.mean_vec)
        stddev_vec = np.array(self.stddev_vec)
        norm_image_data = np.zeros(image_data.shape).astype(np.float32)
        for i in range(image_data.shape[0]):
            norm_image_data[i, :, :] = (image_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
        norm_image_data = norm_image_data.reshape(self.prediction_shape).astype(np.float32)
        return norm_image_data


class SoftmaxTransformer(
    BaseEstimator,
    TransformerMixin,
):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(
        self,
        X: Union[np.ndarray, List[float], List[List[float]]],
    ) -> np.ndarray:
        # 리스트일 경우 np.ndarray로 변환
        if isinstance(X, List):
            X = np.array(X)
        x = X.reshape(-1) # flatten 1-D array로 바꿈
        e_x = np.exp(x - np.max(x)) # exp 값이 비정상적으로 커지는 것을 방지하기 위해 max값 빼기(어차피 상대적 크기에는 영향 x)
        result = np.array([e_x / e_x.sum(axis=0)]) # 2-D array로 전환
        return result


class Classifier(object):
    def __init__(
        self,
        preprocess_transformer: BaseEstimator = PytorchImagePreprocessTransformer,
        softmax_transformer: BaseEstimator = SoftmaxTransformer,
        serving_address: str = "localhost:50051",
        onnx_input_name: str = "input",
        onnx_output_name: str = "output",
    ):
        self.preprocess_transformer: BaseEstimator = preprocess_transformer()
        self.preprocess_transformer.fit(None)
        self.softmax_transformer: BaseEstimator = softmax_transformer()
        self.softmax_transformer.fit(None)

        self.serving_address = serving_address
        self.channel = grpc.insecure_channel(self.serving_address)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        
        self.onnx_input_name: str = onnx_input_name
        self.onnx_output_name: str = onnx_output_name

        


m, 