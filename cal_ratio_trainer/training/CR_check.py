from cal_ratio_trainer.common.fileio import load_dataset
import numpy as np
from cal_ratio_trainer.training.model_input.jet_input import JetInput
from cal_ratio_trainer.training.model_input.model_input import ModelInput
from cal_ratio_trainer.training.training_utils import setup_model_architecture
import matplotlib.pyplot as plt
from cal_ratio_trainer.training.evaluate_training import ks_w2
from keras.models import model_from_json
import logging

def ControlRegion_check(params, cache):
    '''
    MSeg_input, MSeg_input_adversary, constit_input, constit_input_adversary, \
        jet_input, jet_input_adversary, model_to_do, track_input, track_input_adversary = initialize_model(
            params, model_number)
    '''
    print("\nLoading up dataset " + params.cr_training_file + "...\n")
    df_adversary = load_dataset(params.cr_training_file, cache)
    df_adversary = df_adversary.sample(frac=params.frac_list)
    # only use certain jets for plotting
    df_adversary = df_adversary[:int(1e4)]
    
    def modpi(y):
        if y >= 0:
            return y % np.pi
        return -(abs(y) % np.pi)   
    for x in df_adversary.columns:
        if 'phi' in x:
            df_adversary[x] = df_adversary[x].apply(modpi)

    #mass cut
    
    #df_adversary = df_adversary[df_adversary.jet_pt > 0.5]
    #
    
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
    weights_adversary_check = mcWeights_adversary

    ave = np.mean(mcWeights_adversary.values)
    logging.info(f"MC weights average:{ave}")


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

    '''
    weight = np.ones(len(Y_adversary_check))

    qcd_rows = (Y_adversary_check==0)
    data_rows = (Y_adversary_check==1)

    weight[qcd_rows] = weight[qcd_rows]/np.sum(weight[qcd_rows])
    weight[data_rows] = weight[data_rows]/np.sum(weight[data_rows])
    '''
    weights_QCD = mcWeights_adversary[Y_adversary_check==0]
    weights_data = mcWeights_adversary[Y_adversary_check==1]
    weights_QCD = weights_QCD*(Weight_data_sum/Weight_qcd_sum)
    logging.debug(f"QCD weights: {weights_QCD}")
    logging.debug(f"data weights: {weights_data}")
    
    #check
    y_test = Y_adversary_check
    mcWeights_test = weights_adversary_check
    fig, ax = plt.subplots()

    bin_list = np.linspace(0, 1, 30).tolist()
    #bin_list = np.append(bin_list, np.logspace(np.log10(0.00001), np.log10(1.0), 30))
    #bin_list = np.logspace(np.log10(0.00001), np.log10(1.0), 20)
    
    ax.hist(prediction_QCD[:, 2], weights=weights_QCD, color='red', alpha=0.5, linewidth=0,
            histtype='stepfilled', bins=bin_list, label="MC")
    
    n_data, bin_edges_data = np.histogram(prediction_data[:, 2], weights=weights_data,bins=bin_list)
    bin_centers_data = (bin_edges_data[:-1] + bin_edges_data[1:])/2.
    ax.errorbar(bin_centers_data, n_data, fmt='ok', label="data")

    ks = ks_w2(prediction_QCD[:, 2], prediction_data[:, 2], weights_QCD.values, weights_data.values)
    logging.info(ks)

    plt.xlim(0, 1)
    plt.yscale('log')
    #plt.xscale('log')
    
    ax.set_ylabel("Weighted Entries", loc='top')
    ax.set_xlabel("BIB score", loc='right')
    ax.legend(loc='upper right')
    ax.set_title(f"BIB score ks test = {ks}")

    plt.savefig(f"plots/{params.name}_BIB_Score_true_weight.png", format='png', transparent=True)
    plt.clf()
    plt.close(fig)