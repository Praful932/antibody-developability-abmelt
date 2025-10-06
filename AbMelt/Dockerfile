# dockerfile for inference AbMelt rapids
FROM rapidsai/base:25.10a-cuda12-py3.13

USER rapids
WORKDIR /home/rapids

RUN mkdir -p log

COPY models/ ./models/
COPY train_test_predictors/prediction.py ./train_test_predictors/
COPY data/ ./data/

# entrypoint for inference
ENTRYPOINT ["python", "./train_test_predictors/prediction.py"]
# to run you do: docker run abmelt:latest -holdout="data/tmon/holdout.csv" -models="all" -rescaled=0
# haven't ran yet because i don't have rf_rfs.csv, to get that one you need to run     train_test_predictors/rf_rfs.py



