from cal_ratio_trainer.common.fileio import load_dataset
from cal_ratio_trainer.training.model_input.jet_input import JetInput
from cal_ratio_trainer.training.model_input.model_input import ModelInput
from cal_ratio_trainer.training.evaluate_training import ks_w2
from cal_ratio_trainer.training.utils import low_or_high_pt_selection_train
from cal_ratio_trainer.common.column_names import (
    EventType, 
    col_cluster_track_mseg_names,
    col_jet_names,
    col_llp_mass_names,
)

# plotting-umami from flavor tagging, use 'pip install puma-hep' to install
from puma import Histogram, HistogramPlot, Roc, RocPlot
from puma.utils import get_good_linestyles
from puma.metrics import calc_rej

from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import cast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

def control_region_check(params, cache):
    logging.info("Beginning of control region check")
    logging.info("Loading up dataset " + params.cr_training_file + "...")
    df_adversary = load_dataset(params.cr_training_file, cache)
    df_adversary = df_adversary.sample(frac=params.frac_list)
    # df_adversary = df_adversary[:int(1e3)]
    
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

    # load model
    with open(params.json_file1) as json_file1:
        loaded_model_json1 = json_file1.read()
    model1 = model_from_json(loaded_model_json1)
    with open(params.json_file2) as json_file2:
        loaded_model_json2 = json_file2.read()
    model2 = model_from_json(loaded_model_json2)
    # load weights
    model1.load_weights(params.model_weight1) # type: ignore
    model2.load_weights(params.model_weight2) # type: ignore

    prediction1 = model1.predict(X_adversary_check) # type: ignore
    prediction2 = model2.predict(X_adversary_check) # type: ignore
    #split CR to QCD and data
    prediction_QCD1 = prediction1[Y_adversary_check==0]
    prediction_data1 = prediction1[Y_adversary_check==1]
    prediction_QCD2 = prediction2[Y_adversary_check==0]
    prediction_data2 = prediction2[Y_adversary_check==1]
    logging.debug(f"QCD prediction for model1: {prediction_QCD1}")
    logging.debug(f"data prediction for model1: {prediction_data1}")
    logging.debug(f"QCD prediction for model2: {prediction_QCD2}")
    logging.debug(f"data prediction for model2: {prediction_data2}")

    weights_QCD = mcWeights_adversary[Y_adversary_check==0]
    weights_QCD = weights_QCD*(Weight_data_sum/Weight_qcd_sum)
    weights_data = mcWeights_adversary[Y_adversary_check==1]
    logging.debug(f"QCD weights: {weights_QCD}")
    logging.debug(f"data weights: {weights_data}")

    cr_plot(
        params.name1,
        params.name2,
        "no pt cut",
        prediction_QCD1,
        prediction_QCD2,
        weights_QCD,
        prediction_data1,
        prediction_data2,
        weights_data,
    )
    
    def pt_scaled(pt):
        pt_min, pt_max = 40., 500. # in GeV
        return (pt - pt_min) / (pt_max - pt_min)

    index_qcd = df_adversary[Y_adversary_check==0].jet_pt < pt_scaled(120)
    index_data = df_adversary[Y_adversary_check==1].jet_pt < pt_scaled(120)
    cr_plot(
        params.name1,
        params.name2,
        "Jet pt < 120 GeV",
        prediction_QCD1[index_qcd],
        prediction_QCD2[index_qcd],
        weights_QCD[index_qcd],
        prediction_data1[index_data],
        prediction_data2[index_data],
        weights_data[index_data],
    )
    
    index_qcd = (df_adversary[Y_adversary_check==0].jet_pt > pt_scaled(120)) & \
                (df_adversary[Y_adversary_check==0].jet_pt < pt_scaled(250))
    index_data = (df_adversary[Y_adversary_check==1].jet_pt > pt_scaled(120)) & \
                 (df_adversary[Y_adversary_check==1].jet_pt < pt_scaled(250))
    cr_plot(
        params.name1,
        params.name2,
        "120 GeV < Jet pt < 250 GeV",
        prediction_QCD1[index_qcd],
        prediction_QCD2[index_qcd],
        weights_QCD[index_qcd],
        prediction_data1[index_data],
        prediction_data2[index_data],
        weights_data[index_data],
    )
    
    index_qcd = df_adversary[Y_adversary_check==0].jet_pt > pt_scaled(250)
    index_data = df_adversary[Y_adversary_check==1].jet_pt > pt_scaled(250)
    cr_plot(
        params.name1,
        params.name2,
        "Jet pt > 250 GeV",
        prediction_QCD1[index_qcd],
        prediction_QCD2[index_qcd],
        weights_QCD[index_qcd],
        prediction_data1[index_data],
        prediction_data2[index_data],
        weights_data[index_data],
    )


def cr_plot(
        model_name1,
        model_name2,
        plot_label,
        prediction_QCD1,
        prediction_QCD2,
        weights_QCD,
        prediction_data1,
        prediction_data2,
        weights_data,
):
    '''
    Compare BIB score distribution in Control Region.

    Plots will have (0, 1) linear and (1e-6, 1) log-scale x-axis.

    Args:
        model_name1: name of the first model, used to make legend
        model_name2: name of the second model,
        plot_label: label used to specify pt cut, shown in the upper middle of the graph
    '''
    plot_label_dict = {
        "no pt cut": "no_pt_cut",
        "Jet pt > 250 GeV": "pt_gt_250",
        "120 GeV < Jet pt < 250 GeV": "pt_gt_120_lt_250",
        "Jet pt < 120 GeV": "pt_lt_120",
    }

    # ------------------
    # plots with linear x-axis
    # ------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    bin_list = np.linspace(0, 1, 30).tolist()
    
    n_qcd1, _, _ = ax1.hist(prediction_QCD1[:, 2], weights=weights_QCD, color='red', alpha=0.5, linewidth=0,
                histtype='stepfilled', bins=bin_list, density = True, label=f"{model_name1} MC")
    n_data1, bin_edges_data = np.histogram(prediction_data1[:, 2], weights=weights_data, bins=bin_list, density=True)
    bin_centers_data = (bin_edges_data[:-1] + bin_edges_data[1:]) / 2.
    ks1 = ks_w2(prediction_QCD1[:, 2], prediction_data1[:, 2], weights_QCD.values, weights_data.values)
    ax1.errorbar(bin_centers_data, n_data1, fmt='or', label=f"{model_name1} data")
    
    n_qcd2, _, _ = ax1.hist(prediction_QCD2[:, 2], weights=weights_QCD, color='blue', alpha=0.5, linewidth=0,
                histtype='stepfilled', bins=bin_list, density = True, label=f"{model_name2} MC")
    n_data2, bin_edges_data = np.histogram(prediction_data2[:, 2], weights=weights_data, bins=bin_list, density=True)
    bin_centers_data = (bin_edges_data[:-1] + bin_edges_data[1:]) / 2.
    ks2 = ks_w2(prediction_QCD2[:, 2], prediction_data2[:, 2], weights_QCD.values, weights_data.values)
    ax1.errorbar(bin_centers_data, n_data2, fmt='ob', label=f"{model_name2} data")

    ratio1 = np.divide(n_data1, n_qcd1, out=np.zeros_like(n_data1), where=n_qcd1!=0)
    ratio1 = np.where(n_qcd1==0, np.nan, ratio1) # set zero entries to nan
    bin_centers = (np.array(bin_list[:-1]) + np.array(bin_list[1:])) / 2.  
    ax2.plot(bin_centers, ratio1, 'or') 

    ratio2 = np.divide(n_data2, n_qcd2, out=np.zeros_like(n_data2), where=n_qcd2!=0)
    ratio2 = np.where(n_qcd2==0, np.nan, ratio2) # set zero entries to nan
    bin_centers = (np.array(bin_list[:-1]) + np.array(bin_list[1:])) / 2.  
    ax2.plot(bin_centers, ratio2, 'ob') 
    ax2.axhline(1, linestyle='--', linewidth=1)

    ymin, ymax = ax1.get_ylim()
    ymin = 1e-6
    ax1.set_ylim(ymin, ymin * ((ymax/ymin) ** 1.4))

    plt.xlim(0, 1)
    ax1.set_yscale("log")
    ax1.set_ylabel("Events Density", loc='top')
    ax1.legend(loc='upper right')
    ax1.set_title(f"BIB score ks test for {model_name1} model: {ks1}\n BIB score ks test for {model_name2} model: {ks2}")
    plt.text(0.5, 0.9, plot_label, horizontalalignment="center", verticalalignment="center", transform=ax1.transAxes)

    ax2.set_ylabel("data/MC")
    ax2.set_xlabel("BIB score", loc='right')
    ax2.set_ylim(0, 2)

    # white background
    plt.savefig(f"plots/{model_name1}_vs_{model_name2}_{plot_label_dict[plot_label]}_CR_BIB.png", 
                format='png', transparent=False, facecolor='white', bbox_inches='tight')
    plt.clf()
    plt.close(fig)

    # ------------------
    # plots with log x-axis
    # ------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    bin_list = np.logspace(np.log10(1e-6), np.log10(1.0), 30)
    
    n_qcd1, _, _ = ax1.hist(prediction_QCD1[:, 2], weights=weights_QCD, color='red', alpha=0.5, linewidth=0,
                histtype='stepfilled', bins=bin_list, density=True, label=f"{model_name1} MC")
    n_data1, bin_edges_data = np.histogram(prediction_data1[:, 2], weights=weights_data, bins=bin_list, density=True)
    bin_centers_data = (bin_edges_data[:-1] + bin_edges_data[1:]) / 2.
    ks1 = ks_w2(prediction_QCD1[:, 2], prediction_data1[:, 2], weights_QCD.values, weights_data.values)
    ax1.errorbar(bin_centers_data, n_data1, fmt='or', label=f"{model_name1} data")
    
    n_qcd2, _, _ = ax1.hist(prediction_QCD2[:, 2], weights=weights_QCD, color='blue', alpha=0.5, linewidth=0,
                histtype='stepfilled', bins=bin_list, density=True, label=f"{model_name2} MC")
    n_data2, bin_edges_data = np.histogram(prediction_data2[:, 2], weights=weights_data, bins=bin_list, density=True)
    bin_centers_data = (bin_edges_data[:-1] + bin_edges_data[1:]) / 2.
    ks2 = ks_w2(prediction_QCD2[:, 2], prediction_data2[:, 2], weights_QCD.values, weights_data.values)
    ax1.errorbar(bin_centers_data, n_data2, fmt='ob', label=f"{model_name2} data")

    ratio1 = np.divide(n_data1, n_qcd1, out=np.zeros_like(n_data1), where=n_qcd1!=0)
    ratio1 = np.where(n_qcd1==0, np.nan, ratio1) # set zero entries to nan
    bin_centers = (np.array(bin_list[:-1]) + np.array(bin_list[1:])) / 2.  
    ax2.plot(bin_centers, ratio1, 'or') 

    ratio2 = np.divide(n_data2, n_qcd2, out=np.zeros_like(n_data2), where=n_qcd2!=0)
    ratio2 = np.where(n_qcd2==0, np.nan, ratio2) # set zero entries to nan
    bin_centers = (np.array(bin_list[:-1]) + np.array(bin_list[1:])) / 2.  
    ax2.plot(bin_centers, ratio2, 'ob') 
    ax2.axhline(1, linestyle='--', linewidth=1)

    ymin, ymax = ax1.get_ylim()
    ymin = 1e-5
    ax1.set_ylim(ymin, ymin * ((ymax/ymin) ** 1.4))

    plt.xlim(1e-6, 1)
    plt.xscale('log')
    ax1.set_yscale("log")
    ax1.set_ylabel("Events Density", loc='top')
    ax1.legend(loc='upper right')
    ax1.set_title(f"BIB score ks test for {model_name1} model: {ks1}\n BIB score ks test for {model_name2} model: {ks2}")
    plt.text(0.5, 0.9, plot_label, horizontalalignment="center", verticalalignment="center", transform=ax1.transAxes)

    ax2.set_ylabel("data/MC")
    ax2.set_xlabel("BIB score", loc='right')
    ax2.set_ylim(0, 2)

    # white background
    plt.savefig(f"plots/{model_name1}_vs_{model_name2}_{plot_label_dict[plot_label]}_CR_BIB_log.png", 
                format='png', transparent=False, facecolor='white', bbox_inches='tight')
    plt.clf()
    plt.close(fig)


def prediction_check(params, cache):
    '''
    Compare the signal, BIB, QCD prediction distribution for two models using test samples.
    
    Also plot ROC curve of background rejection at (0.2, 1) signal tagging efficiency. 
    '''
    # where to save plots
    dir_name = Path('plots')

    # Load dataset
    logging.info(
        f"Loading main training data from {params.main_training_file}..."
    )
    df = load_dataset(params.main_training_file, cache)
    df = df.sample(frac=params.frac_list)
    # df = df[:int(1e3)]

    # Extract labels    
    Y = df["label"]

    # Pull out the weights for later use
    weights = df["mcEventWeight"]
    mcWeights = df["mcEventWeight"].copy()

    # Rescale the weights so that signal and qcd have the same weight.
    qcd_weight = cast(float, np.sum(mcWeights[Y == 0]))
    sig_weight = cast(float, np.sum(mcWeights[Y == 1]))

    mcWeights.loc[Y == 0] = mcWeights[Y == 0] * (sig_weight / qcd_weight)

    # Build the arrays from an explicit list of columns
    X = df.loc[:, col_cluster_track_mseg_names + col_jet_names + ["eventNumber"]]
    Z = df.loc[:, col_llp_mass_names]

    # Split data into train/test datasets
    random_state = 1
    X_train, X_test, y_train, y_test, weights_train, weights_test, mcWeights_train, mcWeights_test, Z_train, Z_test = \
        train_test_split(X, Y, weights, mcWeights, Z, test_size=0.1, random_state=random_state, shuffle=False)
    X_test, X_val, y_test, y_val, weights_test, weights_val, mcWeights_test, mcWeights_val, Z_test, Z_val = \
        train_test_split(X_test, y_test, weights_test, mcWeights_test, Z_test, test_size=0.5)

    low_mass = params.include_low_mass
    high_mass = params.include_high_mass
    X_train, y_train, Z_train, weights_train, mcWeights_train = low_or_high_pt_selection_train(
        X_train, y_train, weights_train, mcWeights_train, Z_train, low_mass, high_mass
    )

    # setup input layers
    constit_input = ModelInput(
        name="constit",
        rows_max=30,
        num_features=12,
        filters_cnn=params.filters_cnn_constit,
        nodes_lstm=params.nodes_constit_lstm,
        lstm_layers=params.layers_list,
        mH_mS_parametrization=[
            params.mH_parametrization,
            params.mS_parametrization,
        ],
    )
    track_input = ModelInput(
        name="track",
        rows_max=20,
        num_features=10,
        filters_cnn=params.filters_cnn_track,
        nodes_lstm=params.nodes_track_lstm,
        lstm_layers=params.layers_list,
        mH_mS_parametrization=[
            params.mH_parametrization,
            params.mS_parametrization,
        ],
    )
    MSeg_input = ModelInput(
        name="MSeg",
        rows_max=30,
        num_features=6,
        filters_cnn=params.filters_cnn_MSeg,
        nodes_lstm=params.nodes_MSeg_lstm,
        lstm_layers=params.layers_list,
        mH_mS_parametrization=[
            params.mH_parametrization,
            params.mS_parametrization,
        ],
    )
    jet_input = JetInput(
        name="jet",
        num_features=3,
        mH_mS_parametrization=[
            params.mH_parametrization,
            params.mS_parametrization,
        ],
    )


    X_train_constit, X_val_constit, X_test_constit = constit_input.extract_and_split_data(
        X_train, X_val, X_test, Z_train, Z_val, Z_test, "clus_pt_0", "clus_time_"
    )
    X_train_track, X_val_track, X_test_track = track_input.extract_and_split_data(
        X_train, X_val, X_test, Z_train, Z_val, Z_test, "track_pt_0", "track_SCTHits_",
    )
    X_train_MSeg, X_val_MSeg, X_test_MSeg = MSeg_input.extract_and_split_data(
        X_train, X_val, X_test, Z_train, Z_val, Z_test, "MSeg_etaPos_0", "MSeg_t0_",
    )
    X_train_jet, X_val_jet, X_test_jet = jet_input.extract_and_split_data(
        X_train, X_val, X_test, Z_train, Z_val, Z_test, "jet_pt", "jet_phi"
    )
    x_to_test = [X_test_constit, X_test_track, X_test_MSeg, X_test_jet.values]

    # load model
    with open(params.json_file1) as json_file1:
        loaded_model_json1 = json_file1.read()
    model1 = model_from_json(loaded_model_json1)
    with open(params.json_file2) as json_file2:
        loaded_model_json2 = json_file2.read()
    model2 = model_from_json(loaded_model_json2)

    # load weights
    model1.load_weights(params.model_weight1) # type: ignore
    model2.load_weights(params.model_weight2) # type: ignore

    # make predictions
    prediction1 = model1.predict(x_to_test, verbose=0) # type: ignore
    prediction2 = model2.predict(x_to_test, verbose=0) # type: ignore


    plot_prediction_histograms_halfLinear(
        params.name1,
        params.name2,
        dir_name,
        prediction1,
        prediction2,
        y_test,
        mcWeights_test.values,  # type: ignore
        f"{params.name1}_vs_{params.name2}",
        high_mass,
        low_mass,
    )

    plot_prediction_histograms_linear(
        params.name1,
        params.name2,
        dir_name,
        prediction1,
        prediction2,
        y_test,
        mcWeights_test.values,  # type: ignore
        f"{params.name1}_vs_{params.name2}",
        high_mass,
        low_mass,
    )

    plot_ROC_comparison(
        params.name1,
        params.name2,
        dir_name,
        prediction1,
        prediction2,
        y_test,
        mcWeights_test.values,  # type: ignore
        f"{params.name1}_vs_{params.name2}",
        high_mass,
        low_mass,
    )


def plot_prediction_histograms_halfLinear(
    name1: str,
    name2: str,
    destination: Path,
    prediction1: np.ndarray,
    prediction2: np.ndarray,
    labels: pd.Series,
    weight: np.ndarray,
    extra_string: str,
    high_mass: bool,
    low_mass: bool,
):
    sig_rows = np.where(labels == EventType.signal.value)
    qcd_rows = np.where(labels == EventType.QCD.value)
    bib_rows = np.where(labels == EventType.BIB.value)
    plt.clf()
    extra_string = (extra_string + "_") if len(extra_string) > 0 else ""

    atlas_second_tag = "$\\sqrt{s}=13$ TeV, $\\int L = 137$ $fb^{-1}$\n"
    desc_phrase = ""
    if high_mass:
        desc_phrase += "High-$E_T$ Training"
    if low_mass:
        desc_phrase += "Low-$E_T$ Training"
    if high_mass and low_mass:
        desc_phrase = "Combined Training"
    atlas_second_tag += desc_phrase

    weight[qcd_rows] = weight[qcd_rows] / np.sum(weight[qcd_rows])
    weight[sig_rows] = weight[sig_rows] / np.sum(weight[sig_rows])
    weight[bib_rows] = weight[bib_rows] / np.sum(weight[bib_rows])


    linestyles = get_good_linestyles()[:2]

    # signal plot
    sig_plot = HistogramPlot(
        bins = 30,
        bins_range = (0, 1),
        logy = True,
        ylabel = 'Fraction of Events',
        xlabel = 'Signal NN Score',
        atlas_second_tag = atlas_second_tag,
        n_ratio_panels = 1,
        figsize = (6, 4.5),
        ymin_ratio = [0],
    )

    sig_plot.add(
        Histogram(
            prediction1[sig_rows][:, EventType.signal.value],
            label = "Signal",
            colour = "tab:blue",
            linestyle = linestyles[0],
            ratio_group = 'signal',
        ),
        reference = True,
    )    
    sig_plot.add(
        Histogram(
            prediction1[qcd_rows][:, EventType.signal.value],
            label = "SM Multijet",
            colour = "tab:orange",
            linestyle = linestyles[0],
            ratio_group = 'qcd',
        ),
        reference = True,
    )
    sig_plot.add(
        Histogram(
            prediction1[bib_rows][:, EventType.signal.value],
            label = "BIB",
            colour = "tab:green",
            linestyle = linestyles[0],
            ratio_group = 'bib',
        ),
        reference = True,
    )

    sig_plot.add(
        Histogram(
            prediction2[sig_rows][:, EventType.signal.value],
            colour = "tab:blue",
            linestyle = linestyles[1],
            ratio_group = 'signal',
        ),
        reference = False,
    )
    sig_plot.add(
        Histogram(
            prediction2[qcd_rows][:, EventType.signal.value],
            colour = "tab:orange",
            linestyle = linestyles[1],
            ratio_group = 'qcd',
        ),
        reference = False,
    )
    sig_plot.add(
        Histogram(
            prediction2[bib_rows][:, EventType.signal.value],
            colour = "tab:green",
            linestyle = linestyles[1],
            ratio_group = 'bib',
        ),
        reference = False,
    )

    sig_plot.draw()
    sig_plot.make_linestyle_legend(
        linestyles=linestyles,
        labels=[f"{name1} model", f"{name2} model"],
        bbox_to_anchor=(0.6, 1),
    )
    
    sig_plot.savefig(
        f"{destination}/{extra_string}sig_predictions_half_linear.png",
        format='png', transparent=False, facecolor='white', bbox_inches='tight'
    )


    # qcd plot
    qcd_plot = HistogramPlot(
        bins = 30,
        bins_range = (0, 1),
        logy = True,
        ylabel = 'Fraction of Events',
        xlabel = 'SM Multijet NN Score',
        atlas_second_tag = atlas_second_tag,
        n_ratio_panels = 1,
        figsize = (6, 4.5),
        ymin_ratio = [0],
    )

    qcd_plot.add(
        Histogram(
            prediction1[sig_rows][:, EventType.QCD.value],
            label = "Signal",
            colour = "tab:blue",
            linestyle = linestyles[0],
            ratio_group = 'signal',
        ),
        reference = True,
    )    
    qcd_plot.add(
        Histogram(
            prediction1[qcd_rows][:, EventType.QCD.value],
            label = "SM Multijet",
            colour = "tab:orange",
            linestyle = linestyles[0],
            ratio_group = 'qcd',
        ),
        reference = True,
    )
    qcd_plot.add(
        Histogram(
            prediction1[bib_rows][:, EventType.QCD.value],
            label = "BIB",
            colour = "tab:green",
            linestyle = linestyles[0],
            ratio_group = 'bib',
        ),
        reference = True,
    )

    qcd_plot.add(
        Histogram(
            prediction2[sig_rows][:, EventType.QCD.value],
            colour = "tab:blue",
            linestyle = linestyles[1],
            ratio_group = 'signal',
        ),
        reference = False,
    )
    qcd_plot.add(
        Histogram(
            prediction2[qcd_rows][:, EventType.QCD.value],
            colour = "tab:orange",
            linestyle = linestyles[1],
            ratio_group = 'qcd',
        ),
        reference = False,
    )
    qcd_plot.add(
        Histogram(
            prediction2[bib_rows][:, EventType.QCD.value],
            colour = "tab:green",
            linestyle = linestyles[1],
            ratio_group = 'bib',
        ),
        reference = False,
    )

    qcd_plot.draw()
    qcd_plot.make_linestyle_legend(
        linestyles=linestyles,
        labels=[f"{name1} model", f"{name2} model"],
        bbox_to_anchor=(0.6, 1),
    )
    
    qcd_plot.savefig(
        f"{destination}/{extra_string}qcd_predictions_half_linear.png",
        format='png', transparent=False, facecolor='white', bbox_inches='tight'
    )

    
    # bib plot
    bib_plot = HistogramPlot(
        bins = 30,
        bins_range = (0, 1),
        logy = True,
        ylabel = 'Fraction of Events',
        xlabel = 'BIB NN Score',
        atlas_second_tag = atlas_second_tag,
        n_ratio_panels = 1,
        figsize = (6, 4.5),
        ymin_ratio = [0],
    )

    bib_plot.add(
        Histogram(
            prediction1[sig_rows][:, EventType.BIB.value],
            label = "Signal",
            colour = "tab:blue",
            linestyle = linestyles[0],
            ratio_group = 'signal',
        ),
        reference = True,
    )    
    bib_plot.add(
        Histogram(
            prediction1[qcd_rows][:, EventType.BIB.value],
            label = "SM Multijet",
            colour = "tab:orange",
            linestyle = linestyles[0],
            ratio_group = 'qcd',
        ),
        reference = True,
    )
    bib_plot.add(
        Histogram(
            prediction1[bib_rows][:, EventType.BIB.value],
            label = "BIB",
            colour = "tab:green",
            linestyle = linestyles[0],
            ratio_group = 'bib',
        ),
        reference = True,
    )

    bib_plot.add(
        Histogram(
            prediction2[sig_rows][:, EventType.BIB.value],
            colour = "tab:blue",
            linestyle = linestyles[1],
            ratio_group = 'signal',
        ),
        reference = False,
    )
    bib_plot.add(
        Histogram(
            prediction2[qcd_rows][:, EventType.BIB.value],
            colour = "tab:orange",
            linestyle = linestyles[1],
            ratio_group = 'qcd',
        ),
        reference = False,
    )
    bib_plot.add(
        Histogram(
            prediction2[bib_rows][:, EventType.BIB.value],
            colour = "tab:green",
            linestyle = linestyles[1],
            ratio_group = 'bib',
        ),
        reference = False,
    )

    bib_plot.draw()
    bib_plot.make_linestyle_legend(
        linestyles=linestyles,
        labels=[f"{name1} model", f"{name2} model"],
        bbox_to_anchor=(0.6, 1),
    )
    
    bib_plot.savefig(
        f"{destination}/{extra_string}bib_predictions_half_linear.png",
        format='png', transparent=False, facecolor='white', bbox_inches='tight'
    )


def plot_prediction_histograms_linear(
    name1: str,
    name2: str,
    destination: Path,
    prediction1: np.ndarray,
    prediction2: np.ndarray,
    labels: pd.Series,
    weight: np.ndarray,
    extra_string: str,
    high_mass: bool,
    low_mass: bool,
):
    sig_rows = np.where(labels == EventType.signal.value)
    qcd_rows = np.where(labels == EventType.QCD.value)
    bib_rows = np.where(labels == EventType.BIB.value)
    plt.clf()
    extra_string = (extra_string + "_") if len(extra_string) > 0 else ""

    atlas_second_tag = "$\\sqrt{s}=13$ TeV, $\\int L = 137$ $fb^{-1}$\n"
    desc_phrase = ""
    if high_mass:
        desc_phrase += "High-$E_T$ Training"
    if low_mass:
        desc_phrase += "Low-$E_T$ Training"
    if high_mass and low_mass:
        desc_phrase = "Combined Training"
    atlas_second_tag += desc_phrase

    weight[qcd_rows] = weight[qcd_rows] / np.sum(weight[qcd_rows])
    weight[sig_rows] = weight[sig_rows] / np.sum(weight[sig_rows])
    weight[bib_rows] = weight[bib_rows] / np.sum(weight[bib_rows])


    linestyles = get_good_linestyles()[:2]

    # signal plot
    sig_plot = HistogramPlot(
        bins = 30,
        bins_range = (0, 1),
        logy = False,
        ylabel = 'Fraction of Events',
        xlabel = 'Signal NN Score',
        atlas_second_tag = atlas_second_tag,
        n_ratio_panels = 1,
        figsize = (6, 4.5),
        ymin_ratio = [0],
    )

    sig_plot.add(
        Histogram(
            prediction1[sig_rows][:, EventType.signal.value],
            label = "Signal",
            colour = "tab:blue",
            linestyle = linestyles[0],
            ratio_group = 'signal',
        ),
        reference = True,
    )    
    sig_plot.add(
        Histogram(
            prediction1[qcd_rows][:, EventType.signal.value],
            label = "SM Multijet",
            colour = "tab:orange",
            linestyle = linestyles[0],
            ratio_group = 'qcd',
        ),
        reference = True,
    )
    sig_plot.add(
        Histogram(
            prediction1[bib_rows][:, EventType.signal.value],
            label = "BIB",
            colour = "tab:green",
            linestyle = linestyles[0],
            ratio_group = 'bib',
        ),
        reference = True,
    )

    sig_plot.add(
        Histogram(
            prediction2[sig_rows][:, EventType.signal.value],
            colour = "tab:blue",
            linestyle = linestyles[1],
            ratio_group = 'signal',
        ),
        reference = False,
    )
    sig_plot.add(
        Histogram(
            prediction2[qcd_rows][:, EventType.signal.value],
            colour = "tab:orange",
            linestyle = linestyles[1],
            ratio_group = 'qcd',
        ),
        reference = False,
    )
    sig_plot.add(
        Histogram(
            prediction2[bib_rows][:, EventType.signal.value],
            colour = "tab:green",
            linestyle = linestyles[1],
            ratio_group = 'bib',
        ),
        reference = False,
    )

    sig_plot.draw()
    sig_plot.make_linestyle_legend(
        linestyles=linestyles,
        labels=[f"{name1} model", f"{name2} model"],
        bbox_to_anchor=(0.6, 1),
    )
    
    sig_plot.savefig(
        f"{destination}/{extra_string}sig_predictions_linear.png",
        format='png', transparent=False, facecolor='white', bbox_inches='tight'
    )


    # qcd plot
    qcd_plot = HistogramPlot(
        bins = 30,
        bins_range = (0, 1),
        logy = False,
        ylabel = 'Fraction of Events',
        xlabel = 'SM Multijet NN Score',
        atlas_second_tag = atlas_second_tag,
        n_ratio_panels = 1,
        figsize = (6, 4.5),
        ymin_ratio = [0],
    )

    qcd_plot.add(
        Histogram(
            prediction1[sig_rows][:, EventType.QCD.value],
            label = "Signal",
            colour = "tab:blue",
            linestyle = linestyles[0],
            ratio_group = 'signal',
        ),
        reference = True,
    )    
    qcd_plot.add(
        Histogram(
            prediction1[qcd_rows][:, EventType.QCD.value],
            label = "SM Multijet",
            colour = "tab:orange",
            linestyle = linestyles[0],
            ratio_group = 'qcd',
        ),
        reference = True,
    )
    qcd_plot.add(
        Histogram(
            prediction1[bib_rows][:, EventType.QCD.value],
            label = "BIB",
            colour = "tab:green",
            linestyle = linestyles[0],
            ratio_group = 'bib',
        ),
        reference = True,
    )

    qcd_plot.add(
        Histogram(
            prediction2[sig_rows][:, EventType.QCD.value],
            colour = "tab:blue",
            linestyle = linestyles[1],
            ratio_group = 'signal',
        ),
        reference = False,
    )
    qcd_plot.add(
        Histogram(
            prediction2[qcd_rows][:, EventType.QCD.value],
            colour = "tab:orange",
            linestyle = linestyles[1],
            ratio_group = 'qcd',
        ),
        reference = False,
    )
    qcd_plot.add(
        Histogram(
            prediction2[bib_rows][:, EventType.QCD.value],
            colour = "tab:green",
            linestyle = linestyles[1],
            ratio_group = 'bib',
        ),
        reference = False,
    )

    qcd_plot.draw()
    qcd_plot.make_linestyle_legend(
        linestyles=linestyles,
        labels=[f"{name1} model", f"{name2} model"],
        bbox_to_anchor=(0.6, 1),
    )
    
    qcd_plot.savefig(
        f"{destination}/{extra_string}qcd_predictions_linear.png",
        format='png', transparent=False, facecolor='white', bbox_inches='tight'
    )

    
    # bib plot
    bib_plot = HistogramPlot(
        bins = 30,
        bins_range = (0, 1),
        logy = False,
        ylabel = 'Fraction of Events',
        xlabel = 'BIB NN Score',
        atlas_second_tag = atlas_second_tag,
        n_ratio_panels = 1,
        figsize = (6, 4.5),
        ymin_ratio = [0],
    )

    bib_plot.add(
        Histogram(
            prediction1[sig_rows][:, EventType.BIB.value],
            label = "Signal",
            colour = "tab:blue",
            linestyle = linestyles[0],
            ratio_group = 'signal',
        ),
        reference = True,
    )    
    bib_plot.add(
        Histogram(
            prediction1[qcd_rows][:, EventType.BIB.value],
            label = "SM Multijet",
            colour = "tab:orange",
            linestyle = linestyles[0],
            ratio_group = 'qcd',
        ),
        reference = True,
    )
    bib_plot.add(
        Histogram(
            prediction1[bib_rows][:, EventType.BIB.value],
            label = "BIB",
            colour = "tab:green",
            linestyle = linestyles[0],
            ratio_group = 'bib',
        ),
        reference = True,
    )

    bib_plot.add(
        Histogram(
            prediction2[sig_rows][:, EventType.BIB.value],
            colour = "tab:blue",
            linestyle = linestyles[1],
            ratio_group = 'signal',
        ),
        reference = False,
    )
    bib_plot.add(
        Histogram(
            prediction2[qcd_rows][:, EventType.BIB.value],
            colour = "tab:orange",
            linestyle = linestyles[1],
            ratio_group = 'qcd',
        ),
        reference = False,
    )
    bib_plot.add(
        Histogram(
            prediction2[bib_rows][:, EventType.BIB.value],
            colour = "tab:green",
            linestyle = linestyles[1],
            ratio_group = 'bib',
        ),
        reference = False,
    )

    bib_plot.draw()
    bib_plot.make_linestyle_legend(
        linestyles=linestyles,
        labels=[f"{name1} model", f"{name2} model"],
        bbox_to_anchor=(0.6, 1),
    )
    
    bib_plot.savefig(
        f"{destination}/{extra_string}bib_predictions_linear.png",
        format='png', transparent=False, facecolor='white', bbox_inches='tight'
    )


def plot_ROC_comparison(
    name1: str,
    name2: str,
    destination: Path,
    prediction1: np.ndarray,
    prediction2: np.ndarray,
    labels: pd.Series,
    weight: np.ndarray,
    extra_string: str,
    high_mass: bool,
    low_mass: bool,
):
    logging.info("Start of ROC comparison")
    sig_rows = np.where(labels == EventType.signal.value)
    qcd_rows = np.where(labels == EventType.QCD.value)
    bib_rows = np.where(labels == EventType.BIB.value)
    plt.clf()
    extra_string = (extra_string + "_") if len(extra_string) > 0 else ""

    atlas_second_tag = "$\\sqrt{s}=13$ TeV, $\\int L = 137$ $fb^{-1}$\n"
    desc_phrase = ""
    if high_mass:
        desc_phrase += "High-$E_T$ Training"
    if low_mass:
        desc_phrase += "Low-$E_T$ Training"
    if high_mass and low_mass:
        desc_phrase = "Combined Training"
    atlas_second_tag += desc_phrase

    weight[sig_rows] = weight[sig_rows] / np.sum(weight[sig_rows])
    weight[qcd_rows] = weight[qcd_rows] / np.sum(weight[qcd_rows])
    weight[bib_rows] = weight[bib_rows] / np.sum(weight[bib_rows])

    # target signal efficiency
    sig_eff = np.linspace(0.2, 1, 100)

    n_qcd = qcd_rows[0].shape[0]
    n_bib = bib_rows[0].shape[0]
    logging.info(f"Number of QCD is {n_qcd}")
    logging.info(f"Number of BIB is {n_bib}")

    # calculate the rejection
    qcd_rej1 = calc_rej(
        sig_disc=prediction1[:, EventType.signal.value], 
        bkg_disc=prediction1[:, EventType.QCD.value], 
        sig_weights=weight,
        bkg_weights=weight,
        target_eff=sig_eff
        )
    qcd_rej2 = calc_rej(
        sig_disc=prediction2[:, EventType.signal.value], 
        bkg_disc=prediction2[:, EventType.QCD.value], 
        sig_weights=weight,
        bkg_weights=weight,
        target_eff=sig_eff
        )
    bib_rej1 = calc_rej(
        sig_disc=prediction1[:, EventType.signal.value],
        bkg_disc=prediction1[:, EventType.BIB.value],
        sig_weights=weight,
        bkg_weights=weight,
        target_eff=sig_eff
        )
    bib_rej2 = calc_rej(
        sig_disc=prediction2[:, EventType.signal.value],
        bkg_disc=prediction2[:, EventType.BIB.value],
        sig_weights=weight,
        bkg_weights=weight,
        target_eff=sig_eff
    )

    # plot the ROC
    roc_plot = RocPlot(
        n_ratio_panels = 2,
        ylabel = 'Background Rejection',
        xlabel = 'Signal Efficiency',
        atlas_second_tag = atlas_second_tag,
    )

    roc_plot.add_roc(
        Roc(
            sig_eff = sig_eff,
            bkg_rej = qcd_rej1, # type: ignore
            n_test = n_qcd,
            signal_class = 'signal',
            rej_class = 'qcd',
            label = f"{name1} model",
        ),
        reference = True
    )
    roc_plot.add_roc(
        Roc(
            sig_eff = sig_eff,
            bkg_rej = qcd_rej2, # type: ignore
            n_test = n_qcd,
            signal_class = 'signal',
            rej_class = 'qcd',
            label = f"{name2} model",
        ),
        reference = False
    )
    
    roc_plot.add_roc(
        Roc(
            sig_eff = sig_eff,
            bkg_rej = bib_rej1, # type: ignore
            n_test = n_bib,
            signal_class = 'signal',
            rej_class = 'bib',
            label = f"{name1} model",
        ),
        reference = True
    )
    roc_plot.add_roc(
        Roc(
            sig_eff = sig_eff,
            bkg_rej = bib_rej2, # type: ignore
            n_test = n_bib,
            signal_class = 'signal',
            rej_class = 'bib',
            label = f"{name2} model",
            ),
        reference = False,
    )

    roc_plot.set_ratio_class(1, 'qcd')
    roc_plot.set_ratio_class(2, 'bib')
    roc_plot.draw()
    roc_plot.savefig(
        f"{destination}/{extra_string}ROC.png",
        format='png', transparent=False, facecolor='white',
    )