import argparse
import os

import mlflow

def main():
    parser = argparse.ArgumentParser(
        description="Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # parser.add_argument(
    #     "--commit_hash",
    #     type=str,
    #     default="000000",
    #     help="code commit hash",
    # )
    
    parser.add_argument(
        "--preprocess_data",
        type=str,
        default="cifar10",
        help="cifar10 or cifar100; default cifar10",
    )
    
    parser.add_argument(
        "--preprocess_downstream",
        type=str,
        default="./preprocess/data/preprocess",
        help="preprocess downstream directory",
    )

    parser.add_argument(
        "--preprocess_cached_data_id",
        type=str,
        default="",
        help="previous run id for cache",
    )
    
    # parser.add_argument(
    #     "--train_upstream",
    #     type=str,
    #     default="./preprocess/data/preprocess",
    #     help="upstream directory",
    # )
    
    args = parser.parse_args()
    mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))
    
    with mlflow.start_run() as r: # mlflow project를 시작한다. 프로젝트는 local이나 git uri일 수 있다.
        preprocess_run = mlflow.run(
            uri="./preprocess", # uri:local 경로이거나 git repository uri
            entry_point="preprocess",
            backend="local", # local에서 실행 databricks나 kubernetes 옵션 제공
            parameters={
                "data": args.preprocess_data,
                "downstream": args.preprocess_downstream,
                "cached_data_id": args.preprocess_cached_data_id,
            },
        )
        preprocess_run = mlflow.tracking.MlflowClient().get_run(preprocess_run.run_id)
        
        dataset = os.path.join(
            "/tmp/mlruns/",
            str(mlflow_experiment_id),
            preprocess_run.info.run_id,
            "artifacts/downstream_directory",
        )
        
        
        
