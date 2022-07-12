#!/usr/bin/env python 

# Copyright 2021 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this 
# software and associated documentation files (the "Software"), to deal in the Software 
# without restriction, including without limitation the rights to use, copy, modify, 
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
# permit persons to whom the Software is furnished to do so, subject to the following 
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies 
# or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT 
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS IN THE SOFTWARE.

from arml.exp import exp_multiple_attack

# test the basic experiment
file_path = 'data/RML2016.10a_dict.pkl'
# number of cross validation runs 
n_runs = 5
# verbose ? 
verbose = 1
# type of experiment 
scenario = 'WB'
# defenders model 
train_params = {'type': 'vtcnn2', 
                'dropout': 0.5, 
                'val_split': 0.9, 
                'batch_size': 1024, 
                'nb_epoch': 50, 
                'verbose': verbose, 
                'NHWC': [220000, 2, 128, 1],
                'tpu': False, 
                'file_path': 'models/convmodrecnets_CNN2_0.5.wts.h5'}
# adversary's model 
train_adversary_params = {'type': 'vtcnn2', 
                          'dropout': 0.5, 
                          'val_split': 0.9, 
                          'batch_size': 1024, 
                          'nb_epoch': 50, 
                          'verbose': verbose, 
                          'NHWC': [220000, 2, 128, 1],
                          'tpu': False, 
                          'file_path': 'models/convmodrecnets_adversary_CNN2_0.5.wts.h5'}
# run a sequence shift exp?
shift_sequence = False 
# run a sequence shift exp?
shift_amount = 50 
# attack strength
epsilons = [0.00025, 0.0005, 0.001, 0.005, 0.01]
# defense: 
defense = None
adversarial_training = False ### NOT IMPLMENTED  

if shift_sequence:
    # name for the logger
    logger_name = ''.join([
        'aml_radioml_vtcnn2_vtcnn2_scenario_', 
        scenario, 
        '_shift_', 
        str(shift_amount), 
        '_defense_', 
        defense
    ])
    output_path = ''.join([
        'outputs/aml_vtcnn2_vtcnn2_scenario_', 
        scenario, 
        '_radioml_multiple_attack_shift_', 
        str(shift_amount), 
        '_defense_', 
        defense, 
        '.pkl'
    ])
else: 
    # name for the logger
    logger_name = ''.join([
        'aml_radioml_vtcnn2_vtcnn2_scenario_', 
        scenario, 
        '_defense_', 
        defense
    ])
    output_path = ''.join([
        'outputs/aml_vtcnn2_vtcnn2_scenario_', 
        scenario, 
        '_radioml_multiple_attack', 
        '_defense_', 
        defense, 
        '.pkl'
    ])



experiment_adversarial(file_path=file_path,
                       n_runs=n_runs, 
                       verbose=verbose, 
                       epsilon=epsilon, 
                       scenario=scenario, 
                       train_params=train_params, 
                       shift_sequence=shift_sequence, 
                       shift_amount=shift_amount, 
                       train_adversary_params=train_adversary_params, 
                       adversarial_training=adversarial_training, 
                       defense=defense,
                       epsilons=epsilons, 
                       logger_name:str=logger_name,
                       output_path=output_path)

