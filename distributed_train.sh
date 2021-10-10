#!/bin/bash
NUM_PROC=$1
RANDOM=1355
shift
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$RANDOM train.py "$@"
