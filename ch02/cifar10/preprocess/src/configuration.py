import os
from logging import getLogger

from src.constants import CONSTANTS, PLATFORM_ENUM


logger = getLogger(__name__)


class PlatformConfigurations:
    platform = os.getenv("PLATFORM", PLATFORM_ENUM.DOCKER.value)
    # 값이 없다면 에러 발생
    if not PLATFORM_ENUM.has_value(platform):
        raise ValueError(f"PLATFORM must be one of {[v.value for v in PLATFORM_ENUM.__members__.values()]}")


class PreprocessConfigurations:
    # 다운로드한 학습 데이터 경로
    train_files = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]
    # 다운로드한 테스트 데이터 경로
    test_files = ["test_batch"]

    # 클래스와 레이블
    classes = {
        0: "plane",
        1: "car",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

    
class ModelConfigurations:
    pass


logger.info(f"{PlatformConfigurations.__name__}: {PlatformConfigurations.__dict__}")
logger.info(f"{PreprocessConfigurations.__name__}: {PreprocessConfigurations.__dict__}")
logger.info(f"{ModelConfigurations.__name__}: {ModelConfigurations.__dict__}")
