# Test Plan

## Goal
Goal: Train N UNet SNGP models to compare how it preforms to the base UNet and the hyper parameters affect performance in training and inference. Additionally, compare to more traditional UQ methods such as MC Drouput.

### Code to Add
- Unified metrics .py document
- training metrics logger and csv writer
- test metrics logger and csv writer
- inference (images) .py
- unified study code
    - calls and creates all cases and submits to ARC jobs
- have a function to create the directory structure to ensure they are all there (?)

## Metrics
A (?) indicates I am not familiar with that metric
- Segmentation
    1. mIoU
    2. Dice
    3. Pixelwise Accuracy

- Calibration: quantify how well predicted confidence aligns w/ the true accuracy.
    1. ECE
    2. MCE
    3. Brier    (?)
    4. NLL      (?)
    5. Reliability Diagram  (?)

- OOD: ability to detect anamolous datad
    1. AUROC        (?)
    2. AUPR         (?)
    3. FPR @ 95TPR  (?)

- Efficiency
    1. Step time
    2. Thouroughput
    3. GPU Memory Usage

- Training
    1. Val & Training Loss
    2. Val & train accuracy & Dice score

## Directory Structure
### For the entire test
Currently I am not sure if I will want to have this all fill in with a single .slurm or .py. Maybe break it up so each test case, i.e.

|
| - runs
    |-runs_manifest.csv # (includes info needed to get to the folder; so I don't have to walk through folders)(?)
    |
    |-Unet_control
    |    |- run_1
    |    .
    |    |- run_n
    |
    |- UNet_SNGP
        |
        |-Base_hypers
        |   |-run_1
        |   .
        |   |-run_n
        |
        |-RFF_dim
        |    |-dim_64
        |    .
        |    |-dim_n
        |        |-run_1
        |        .
        |        |-run_n
        |
        |-Proj_dim
        |   |-dim_64
        |   .
        |    |-dim_n
        |        |-run_1
        |        .
        |        |-run_n
        |        
        |-Ridge_penalty
        |   |-pen_1
            .
            |-pen_n
                |-run_1
                .
                |-run_n

### What is in each *run_n*
-run_n
    |-config.yaml # have all hypers, dataset infto, slurm job id and seed
    |-model.pth.tar
    |-tb_file
    |-(all tb action files)
    |-training_metrics.csv
    |-test_metrics.csv
    |-test_images
        |-IID
        |   |-...
        |
        |-OOD
            |-...


## Questions to Answer
- What metrics are actually important?
- How mant runs are needed to get a proper confidence interval?
- How am I going to store these runs (answered above)