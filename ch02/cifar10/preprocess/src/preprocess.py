import argparse
import json
import os
from distutils.dir_util import copy_tree

import mlflow
import torchvision
from src.configurations import PreprocessConfigurations
from src.extract_data import parse_pickle, unpickle


def main():
    parser = argparse.ArgumentParser(
        description="Make dataset",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    # 데이터 이름
    parser.add_argument(
        "--data",
        type=str,
        default="cifar10",
        help="cifar10 or cifar100; default cifar10",
    )
    
    # 데이터 파일을 저장할 경로
    parser.add_argument(
        "--downstream",
        type=str,
        default="/opt/cifar10/preprocess/",
        help="downstream directory",
    )
    
    # cache를 위한 이전 run id
    parser.add_argument(
        "--cached_data_id",
        type=str,
        default="",
        help="previous run id for cache",
    )
    
    args = parser.parse_args()
    downstream_directory = args.downstream
    
    # 이전 cached data가 있다면 붙여넣기
    if args.cached_data_id:
        cached_artifact_directory = os.path.join(
            "/tmp/mlruns/0",
            args.cached_data_id,
            "artifacts/downstream_directory",
        )
        copy_tree(
            cached_artifact_directory,
            downstream_directory,
        )
    
    else:
        # train과 test 데이터 구분
        train_output_destination = os.path.join(
            downstream_directory,
            "train",
        )
        test_output_destination = os.path.join(
            downstream_directory,
            "test",
        )
        cifar10_directory = os.path.join(
            downstream_directory,
            "cifar-10-batches-py",
        )
        
        os.makedirs(downstream_directory, exist_ok=True)
        os.makedirs(train_output_destination, exist_ok=True)
        os.makedirs(test_output_destination, exist_ok=True)
        os.makedirs(cifar10_directory, exist_ok=True)
        
        # root/cifar-10-batches-py 디렉토리에 cifar-10-python.tar.gz 데이터 다운로드 한 후
        # data_batch_{num} 형태로 extract
        torchvision.datasets.CIFAR10(
            root=downstream_directory,
            train=True,
            download=True,
        )
        
        # root/cifar-10-batches-py 디렉토리에 cifar-10-python.tar.gz 데이터 다운로드 한 후
        # test_batch 형태로 extract
        torchvision.datasets.CIFAR10(
            root=downstream_directory,
            train=False,
            download=True,
        )
        
        # 각 클래스별 파일 목록을 담는 딕셔너리
        meta_train = {i: [] for i in range(10)}
        meta_test = {i: [] for i in range(10)}
        
        
        # train dataset 다운
        for f in PreprocessConfigurations.train_files:
            rawdata = unpickle(file=os.path.join(cifar10_directory, f))
            class_to_filename = parse_pickle(
                rawdata=rawdata,
                rootdir=train_output_destination,
            ) # rawdata의 파일을 train_output_destination / 클래스 레이블 / 디렉토리에 저장
            for cf in class_to_filename: # clas_to_filename: cf: ([label, filname]) 를 담은 리스트
                meta_train[int(cf[0])].append(cf[1]) # cf: [label, filname]
        
        # test_set
        for f in PreprocessConfigurations.test_files:
            rawdata = unpickle(file=os.path.join(cifar10_directory, f))
            class_to_filename = parse_pickle(
                rawdata=rawdata,
                rootdir=test_output_destination,
            )
            for cf in class_to_filename:
                meta_test[int(cf[0])].append(cf[1])
        
        classes_filepath = os.path.join(
            downstream_directory,
            "classes.json",
        )
        
        meta_train_filepath = os.path.join(
            downstream_directory,
            "meta_train.json",
        )
        
        meta_test_filepath = os.path.join(
            downstream_directory,
            "meta_test.json",
        )
        
        with open(classes_filepath, "w") as f:
            json.dump(PreprocessConfigurations.classes, f)
        with open(meta_train_filepath, "w") as f:
            json.dump(meta_train, f)
        with open(meta_test_filepath, "w") as f:
            json.dump(meta_test, f)
        
    
    # ?? downstream_directory와 artifact_path에 로그를 저장
    mlflow.log_artifacts(
        downstream_directory,
        artifact_path="downstream_directory",
    )

if __name__ == "__main__":
    main()
