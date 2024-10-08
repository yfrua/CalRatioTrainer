from cal_ratio_trainer.common.fileio import load_dataset
from cal_ratio_trainer.training.model_input.jet_input import JetInput
from cal_ratio_trainer.training.model_input.model_input import ModelInput
from cal_ratio_trainer.training.evaluate_training import ks_w2

import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import logging

def ControlRegion_check(params, cache):
    '''
    MSeg_input, MSeg_input_adversary, constit_input, constit_input_adversary, \
        jet_input, jet_input_adversary, model_to_do, track_input, track_input_adversary = initialize_model(
            params, model_number)
    '''
    logging.info("Loading up dataset " + params.cr_training_file + "...")
    df_adversary = load_dataset(params.cr_training_file, cache)
    df_adversary = df_adversary.sample(frac=params.frac_list)
    
    def modpi(y):
        if y >= 0:
            return y % np.pi
        return -(abs(y) % np.pi)   
    for x in df_adversary.columns:
        if 'phi' in x:
            df_adversary[x] = df_adversary[x].apply(modpi)

    Y_adversary = df_adversary['label']
    Y_adversary[Y_adversary == 2] = 1 # This line gives an error but as far as I can tell is still working as desired
    Y_adversary = np.array(Y_adversary.values) 

    mcWeights_adversary = df_adversary['mcEventWeight']
    X_adversary = df_adversary.loc[:, 'clus_pt_0':'MSeg_t0_29']
    X_adversary = df_adversary.loc[:, 'jet_pt':'jet_phi'].join(X_adversary)
    X_adversary['eventNumber'] = df_adversary['eventNumber']
    Z_adversary = df_adversary.loc[:, 'jet_pt':'jet_eta']

    Weight_qcd_sum= np.sum(mcWeights_adversary[Y_adversary==0])
    Weight_data_sum= np.sum(mcWeights_adversary[Y_adversary==1])


    constit_input_adversary = ModelInput(name='constit', rows_max=30, num_features=12,
                                         filters_cnn=params.filters_cnn_constit,
                                         nodes_lstm=params.nodes_constit_lstm,
                                         lstm_layers=params.layers_list,
                                         mH_mS_parametrization=[params.mH_parametrization,
                                                                params.mS_parametrization])
    track_input_adversary = ModelInput(name='track', rows_max=20, num_features=10,
                                       filters_cnn=params.filters_cnn_track,
                                       nodes_lstm=params.nodes_track_lstm,
                                       lstm_layers=params.layers_list,
                                       mH_mS_parametrization=[params.mH_parametrization,
                                                              params.mS_parametrization])
    MSeg_input_adversary = ModelInput(name='MSeg', rows_max=30, num_features=6,
                                      filters_cnn=params.filters_cnn_MSeg,
                                      nodes_lstm=params.nodes_MSeg_lstm,
                                      lstm_layers=params.layers_list,
                                      mH_mS_parametrization=[params.mH_parametrization,
                                                             params.mS_parametrization])
    jet_input_adversary = JetInput(name='jet', num_features=3,
                                   mH_mS_parametrization=[params.mH_parametrization, params.mS_parametrization])

    X_constit_adversary, _ ,_ = constit_input_adversary.extract_and_split_data(
        X_adversary, X_adversary, X_adversary,
        Z_adversary, Z_adversary, Z_adversary,
        'clus_pt_0', 'clus_time_')
    
    X_track_adversary, _,_ = track_input_adversary.extract_and_split_data(
        X_adversary, X_adversary, X_adversary,
        Z_adversary, Z_adversary, Z_adversary,
        'track_pt_0', 'track_SCTHits_')   
    
    X_MSeg_adversary, _,_ = MSeg_input_adversary.extract_and_split_data(
        X_adversary, X_adversary, X_adversary,
        Z_adversary, Z_adversary, Z_adversary,
        'MSeg_etaPos_0', 'MSeg_t0_')   
    
    X_jet_adversary, _,_= jet_input_adversary.extract_and_split_data(        
        X_adversary, X_adversary, X_adversary,
        Z_adversary, Z_adversary, Z_adversary,
        'jet_pt', 'jet_phi')    

    X_adversary_check = [X_constit_adversary, X_track_adversary, X_MSeg_adversary, X_jet_adversary.values]
    Y_adversary_check = Y_adversary

    #load model
    json_file = open(params.json_file)
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #load weights
    model.load_weights(params.model_weight)


    prediction = model.predict(X_adversary_check)
    #split CR to QCD and data
    prediction_QCD = prediction[Y_adversary_check==0]
    prediction_data = prediction[Y_adversary_check==1]
    logging.debug(f"QCD prediction: {prediction_QCD}")
    logging.debug(f"data prediction: {prediction_data}")

    weights_QCD = mcWeights_adversary[Y_adversary_check==0]
    weights_data = mcWeights_adversary[Y_adversary_check==1]
    weights_QCD = weights_QCD*(Weight_data_sum/Weight_qcd_sum)
    logging.debug(f"QCD weights: {weights_QCD}")
    logging.debug(f"data weights: {weights_data}")

    _plot(
        params.name,
        "no pt cut",
        prediction_QCD,
        weights_QCD,
        prediction_data,
        weights_data,
    )
    
    index_qcd = df_adversary[Y_adversary_check==0].jet_pt < _pt_scaled(120)
    index_data = df_adversary[Y_adversary_check==1].jet_pt < _pt_scaled(120)
    _plot(
        params.name,
        "Jet pt < 120 GeV",
        prediction_QCD[index_qcd],
        weights_QCD[index_qcd],
        prediction_data[index_data],
        weights_data[index_data],
    )
    
    index_qcd = (df_adversary[Y_adversary_check==0].jet_pt > _pt_scaled(120)) & \
                (df_adversary[Y_adversary_check==0].jet_pt < _pt_scaled(250))
    index_data = (df_adversary[Y_adversary_check==1].jet_pt > _pt_scaled(120)) & \
                 (df_adversary[Y_adversary_check==1].jet_pt < _pt_scaled(250))
    _plot(
        params.name,
        "120 GeV < Jet pt < 250 GeV",
        prediction_QCD[index_qcd],
        weights_QCD[index_qcd],
        prediction_data[index_data],
        weights_data[index_data],
    )
    
    index_qcd = df_adversary[Y_adversary_check==0].jet_pt > _pt_scaled(250)
    index_data = df_adversary[Y_adversary_check==1].jet_pt > _pt_scaled(250)
    _plot(
        params.name,
        "Jet pt > 250 GeV",
        prediction_QCD[index_qcd],
        weights_QCD[index_qcd],
        prediction_data[index_data],
        weights_data[index_data],
    )


plot_label_dict = {
    "no pt cut": "no_pt_cut",
    "Jet pt > 250 GeV": "pt_gt_250",
    "120 GeV < Jet pt < 250 GeV": "pt_gt_120_lt_250",
    "Jet pt < 120 GeV": "pt_lt_120",
}

def _pt_scaled(pt):
    pt_min, pt_max = 40, 500 # in GeV
    return (pt - pt_min) / (pt_max - pt_min)

def _plot(
        model_name,
        plot_label,
        prediction_QCD,
        weights_QCD,
        prediction_data,
        weights_data,
):
    # ------------------
    # plots with linear x-axis
    # ------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    bin_list = np.linspace(0, 1, 30).tolist()
    
    n_qcd, _, _ = ax1.hist(prediction_QCD[:, 2], weights=weights_QCD, color='red', alpha=0.5, linewidth=0,
                histtype='stepfilled', bins=bin_list, label="MC")

    n_data, bin_edges_data = np.histogram(prediction_data[:, 2], weights=weights_data, bins=bin_list)
    bin_centers_data = (bin_edges_data[:-1] + bin_edges_data[1:]) / 2.
    ax1.errorbar(bin_centers_data, n_data, fmt='ok', label="data")

    ks = ks_w2(prediction_QCD[:, 2], prediction_data[:, 2], weights_QCD.values, weights_data.values)

    ratio = np.divide(n_data, n_qcd, out=np.zeros_like(n_data), where=n_qcd!=0)
    ratio = np.where(n_qcd==0, np.nan, ratio) # set zero entries to nan
    bin_centers = (np.array(bin_list[:-1]) + np.array(bin_list[1:])) / 2.  
    ax2.plot(bin_centers, ratio, 'ok') 
    ax2.axhline(1, linestyle='--', linewidth=1)

    plt.xlim(0, 1)
    ax1.set_yscale("log")
    ax1.set_ylabel("Weighted Entries", loc='top')
    ax1.legend(loc='upper right')
    ax1.set_title(f"BIB score ks test = {ks}")
    plt.text(0.7, 0.9, plot_label, horizontalalignment="center", verticalalignment="center", transform=ax1.transAxes)

    ax2.set_ylabel("data/MC")
    ax2.set_xlabel("BIB score", loc='right')
    ax2.set_ylim(0, 2)

    # white background
    plt.savefig(f"plots/{model_name}_{plot_label_dict[plot_label]}_BIB_Score_true_weight.png", format='png', transparent=False, facecolor='white', bbox_inches='tight')
    # transparent background
    # plt.savefig(f"plots/{model_name}_{plot_label_dict[plot_label]}_BIB_Score_true_weight.png", format='png', transparent=True)
    plt.clf()
    plt.close(fig)

    # ------------------
    # plots with log x-axis
    # ------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    bin_list = np.logspace(np.log10(1e-6), np.log10(1.0), 30)
    
    n_qcd, _, _ = ax1.hist(prediction_QCD[:, 2], weights=weights_QCD, color='red', alpha=0.5, linewidth=0,
                histtype='stepfilled', bins=bin_list, label="MC")

    n_data, bin_edges_data = np.histogram(prediction_data[:, 2], weights=weights_data, bins=bin_list)
    bin_centers_data = (bin_edges_data[:-1] + bin_edges_data[1:]) / 2.
    ax1.errorbar(bin_centers_data, n_data, fmt='ok', label="data")

    ks = ks_w2(prediction_QCD[:, 2], prediction_data[:, 2], weights_QCD.values, weights_data.values)

    ratio = np.divide(n_data, n_qcd, out=np.zeros_like(n_data), where=n_qcd!=0)
    ratio = np.where(n_qcd==0, np.nan, ratio) # set zero entries to nan
    bin_centers = (np.array(bin_list[:-1]) + np.array(bin_list[1:])) / 2.  
    ax2.plot(bin_centers, ratio, 'ok') 
    ax2.axhline(1, linestyle='--', linewidth=1)

    plt.xlim(1e-6, 1)
    plt.xscale('log')
    ax1.set_yscale("log")
    ax1.set_ylabel("Weighted Entries", loc='top')
    ax1.legend(loc='upper right')
    ax1.set_title(f"BIB score ks test = {ks}")
    plt.text(0.7, 0.9, plot_label, horizontalalignment="center", verticalalignment="center", transform=ax1.transAxes)

    ax2.set_ylabel("data/MC")
    ax2.set_xlabel("BIB score", loc='right')
    ax2.set_ylim(0, 2)

    # white background
    plt.savefig(f"plots/{model_name}_{plot_label_dict[plot_label]}_BIB_Score_true_weight_log.png", format='png', transparent=False, facecolor='white', bbox_inches='tight')
    # transparent background
    # plt.savefig(f"plots/{model_name}_{plot_label_dict[plot_label]}_BIB_Score_true_weight_log.png", format='png', transparent=True)
    plt.clf()
    plt.close(fig)