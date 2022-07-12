#!/usr/bin/env python 

# Copyright 2022
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

import pickle 
import numpy as np 

from copy import copy 
from ..utils import load_radioml
from ..models import nn_model
from ..performance import AdversarialPerfLogger, PerfLogger, FGSMPerfLogger, AdversarialDefenseLogger
from ..adversarial_data import generate_aml_data

from sklearn.model_selection import KFold

from art.defences.postprocessor import GaussianNoise, ClassLabels, HighConfidence, ReverseSigmoid

import neural_structured_learning as nsl


def experiment_adversarial(file_path:str,
                           n_runs:int=5, 
                           verbose:int=1, 
                           scenario:str='A', 
                           train_params:dict={}, 
                           shift_sequence:bool=True, 
                           shift_amount:int=50, 
                           train_adversary_params:dict={}, 
                           adversarial_training:bool=False, 
                           defense:str=None, 
                           epsilons:list=[0.00025, 0.0005, 0.001, 0.005, 0.01], 
                           logger_name:str='aml_radioml_vtcnn2_vtcnn2_scenario_A',
                           output_path:str='outputs/aml_vtcnn2_vtcnn2_scenario_A_radioml.pkl'): 
    """run mulltiple attacks (FGSM, DeepFool, PGD) on the radioml dataset. note that this is not going to run 
    on google colab because it takes too long to execute. must be run on another cloud service or workstation. 

    Parameters
    ---------- 
    file_path : str
        Location of the radioml dataset
    n_runs : int
        Number of cross validations  
    verbose : int
        Verbose?  
    epsilons : list
        attack strength
    shift_sequence : bool
        do you want to shift the location of the perturbation 
    shift_amount : int
        how much do you shift the perturbation 
    scenario : str 
        Adversary knowledge: 
            'GB': has an NN structure and a subset of the training data  
    train_params : dict
        Training parameters
            train_params = {'type': 'vtcnn2', 
                        'dropout': 0.5, 
                        'val_split': 0.9, 
                        'batch_size': 1024, 
                        'nb_epoch': 50, 
                        'verbose': verbose, 
                        'NHWC': [N, H, W, C],
                        'tpu': False, 
                        'file_path': 'convmodrecnets_CNN2_0.5.wts.h5'}
    train_adversary_params : dict
        Training parameters 
            train_adversary_params = {'type': 'vtcnn2', 
                                  'dropout': 0.5, 
                                  'val_split': 0.9, 
                                  'batch_size': 1024, 
                                  'nb_epoch': 50, 
                                  'verbose': verbose, 
                                  'NHWC': [N, H, W, C],
                                  'epsilon': 0.15, 
                                  'file_path': 'convmodrecnets_adversary_CNN2_0.5.wts.h5'}
    logger_name : str
        Name of the logger class [default: 'aml_radioml_vtcnn2_vtcnn2_scenario_A']
    output_path : str
        Output path [default: outputs/aml_vtcnn2_vtcnn2_scenario_A_radioml.pkl]
    """

    X, Y, snrs, mods, _ = load_radioml(file_path=file_path, shuffle=True)
    C = 1
    N, H, W = X.shape
    X = X.reshape(N, H, W, C)

    if len(train_params) == 0:
        train_params = {'type': 'vtcnn2', 
                        'dropout': 0.5, 
                        'val_split': 0.9, 
                        'batch_size': 1024, 
                        'nb_epoch': 50, 
                        'verbose': verbose, 
                        'NHWC': [N, H, W, C],
                        'tpu': False, 
                        'file_path': 'models/convmodrecnets_CNN2_0.5.wts.h5'}
    
    if len(train_adversary_params) == 0:
        train_adversary_params = {'type': 'vtcnn2', 
                                  'dropout': 0.5, 
                                  'val_split': 0.9, 
                                  'batch_size': 1024, 
                                  'nb_epoch': 50, 
                                  'verbose': verbose, 
                                  'NHWC': [N, H, W, C],
                                  'file_path': 'models/convmodrecnets_adversary_CNN2_0.5.wts.h5'}
    
    # initialize the performances to empty 
    # logger_with_epsilon = {'type', np.unique(snrs), np.unique(mods), [train_params, train_adversary_params], epsilons}
    # logger_deepfool = {'type', np.unique(snrs), np.unique(mods), [train_params, train_adversary_params]}

    result_fgsm_logger = FGSMPerfLogger('type', np.unique(snrs), np.unique(mods), [train_params, train_adversary_params], epsilons)
    result_pgd_logger = FGSMPerfLogger('type', np.unique(snrs), np.unique(mods), [train_params, train_adversary_params], epsilons)
    result_deepfool_logger = AdversarialPerfLogger('type', np.unique(snrs), np.unique(mods), [train_params, train_adversary_params])
    result_fgsm_defense_logger = AdversarialDefenseLogger('type', np.unique(snrs), np.unique(mods), [train_params, train_adversary_params], epsilons)
    result_pgd_defense_logger = AdversarialDefenseLogger('type', np.unique(snrs), np.unique(mods), [train_params, train_adversary_params], epsilons)
    result_deepfool_defense_logger = AdversarialDefenseLogger('type', np.unique(snrs), np.unique(mods), [train_params, train_adversary_params], [0])

    
    kf = KFold(n_splits=n_runs)
    
    for train_index, test_index in kf.split(X): 
        # split out the training and testing data. do the sample for the modulations and snrs
        Xtr, Ytr, Xte, Yte, snrs_te = X[train_index], Y[train_index], X[test_index], Y[test_index], snrs[test_index]
        Xte, Yte, snrs_te = Xte[:1000], Yte[:1000], snrs_te[:1000]
        if verbose: 
            print('Training the defenders model')
        model = nn_model(X=Xtr, Y=Ytr, train_param=train_params, adversarial_training=adversarial_training) 

        if scenario == 'GB': 
            # sample adversarial training data 
            Ntr = len(Xtr)
            sample_indices = np.random.randint(0, Ntr, Ntr)        
            # train the model
            model_aml = nn_model(X=Xtr[sample_indices], Y=Ytr[sample_indices], train_param=train_adversary_params) 
        elif scenario == 'WB': # completely whitebox 
            model_aml = copy(model)
        
        postprocessor_gn = GaussianNoise(scale=0.1)
        postprocessor_cl = ClassLabels()
        postprocessor_hc = HighConfidence(cutoff=0.1)
        postprocessor_rc = ReverseSigmoid(beta=1.0, gamma=0.1)
        
        # evaluate Deepfool 
        if verbose: 
            print('Running DeepFool')
        Xdeep = generate_aml_data(model_aml, Xte, Yte, {'type': 'DeepFool'})
        if shift_sequence: 
            # get the perturbation 
            delta_deep = Xdeep - Xte
            # apply the shift 
            delta_deep[:, :, :shift_amount, 0] = 0
            Xdeep = Xte + delta_deep
        
        # for each of the snrs -> grab all of the data for that snr, which should have all of
        # the classes then evaluate the model on the data for the snr under test. store the 
        # aucs, accs, and ppls in a dictionary 
        for snr in np.unique(snrs_te): 
            Yhat = model.predict(Xte[snrs_te == snr]) 
            Yhat_deep = model.predict(Xdeep[snrs_te == snr])

            result_deepfool_defense_logger.add_scores(
                Yte[snrs_te==snr], 
                postprocessor_gn(Yhat_deep), 
                postprocessor_cl(Yhat_deep), 
                postprocessor_hc(Yhat_deep), 
                postprocessor_rc(Yhat_deep), 
                snr, 
                0
            )
            result_deepfool_logger.add_scores(Yte[snrs_te==snr], Yhat, Yhat, Yhat_deep, Yhat, snr)
        
        
        # do fgsm and pgd 
        # loop through the different values of epsilon and generate adversarial datasets
        for eps_index, eps in enumerate(epsilons): 
            if verbose: 
                print(''.join(['Running PGD and FGSM with epsilon=', str(eps)]))
            Xfgsm = generate_aml_data(model_aml, Xte, Yte, {'type': 'FastGradientMethod', 'eps': eps})
            Xpgd = generate_aml_data(model_aml, Xte, Yte, {'type': 'ProjectedGradientDescent', 'eps': eps, 
                                                           'eps_step':0.0005, 'max_iter': 50})

            if shift_sequence: 
                # get the perturbation 
                delta_fgsm, delta_pgd = Xfgsm - Xte, Xpgd - Xte
                # apply the shift 
                delta_fgsm[:, :, :shift_amount, 0] = 0
                delta_pgd[:, :, :shift_amount, 0] = 0
                Xfgsm = Xte + delta_fgsm
                Xpgd = Xte + delta_pgd

            for snr in np.unique(snrs_te): 
                Yhat_fgsm = model.predict(Xfgsm[snrs_te == snr])
                Yhat_pgd = model.predict(Xpgd[snrs_te == snr])

                result_fgsm_defense_logger.add_scores(
                    Yte[snrs_te==snr], 
                    postprocessor_gn(Yhat_fgsm), 
                    postprocessor_cl(Yhat_fgsm), 
                    postprocessor_hc(Yhat_fgsm), 
                    postprocessor_rc(Yhat_fgsm), 
                    snr, 
                    eps_index
                )
                result_pgd_defense_logger.add_scores(
                    Yte[snrs_te==snr], 
                    postprocessor_gn(Yhat_pgd), 
                    postprocessor_cl(Yhat_pgd), 
                    postprocessor_hc(Yhat_pgd), 
                    postprocessor_rc(Yhat_pgd), 
                    snr, 
                    eps_index
                )

                result_pgd_logger.add_scores(Yte[snrs_te==snr], Yhat_pgd, snr, eps_index)
                result_fgsm_logger.add_scores(Yte[snrs_te==snr], Yhat_fgsm, snr, eps_index)

        

        # save the results to a pickle file 
        results = {
            'result_fgsm_logger': result_fgsm_logger, 
            'result_pgd_logger': result_pgd_logger, 
            'result_deepfool_logger': result_deepfool_logger, 
            'result_fgsm_defense_logger': result_fgsm_defense_logger, 
            'result_pgd_defense_logger': result_pgd_defense_logger, 
            'result_deepfool_defense_logger': result_deepfool_defense_logger  
        }
        pickle.dump(results, open(output_path, 'wb'))

        
    result_pgd_logger.finalize()
    result_fgsm_logger.finalize()
    result_deepfool_logger.finalize()
    result_fgsm_defense_logger.finalize()
    result_pgd_defense_logger.finalize()
    result_deepfool_defense_logger.finalize()

    # save the results to a pickle file 
    results = {
        'result_fgsm_logger': result_fgsm_logger, 
        'result_pgd_logger': result_pgd_logger, 
        'result_deepfool_logger': result_deepfool_logger, 
        'result_fgsm_defense_logger': result_fgsm_defense_logger, 
        'result_pgd_defense_logger': result_pgd_defense_logger, 
        'result_deepfool_defense_logger': result_deepfool_defense_logger  
    }
    pickle.dump(results, open(output_path, 'wb'))