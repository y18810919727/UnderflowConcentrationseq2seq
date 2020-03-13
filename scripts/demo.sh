#!/usr/bin/env bash

cd ..
python main.py --use_cuda --hidden_num 32 --net_type rnn --epochs 100 --random_seed 98901 --save_dir _ode_4_5 --algorithm ode

