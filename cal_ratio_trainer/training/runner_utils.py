import logging
from typing import Tuple

import numpy as np

from cal_ratio_trainer.config import TrainingConfig
from cal_ratio_trainer.training.deepJet_train_keras import train_llp
from cal_ratio_trainer.training.model_input.jet_input import JetInput
from cal_ratio_trainer.training.model_input.model_input import ModelInput


def training_runner_util(training_parameters: TrainingConfig):
    """Run the training for the LLP neural network."""

    # initialize the model
    (
        MSeg_input,
        MSeg_input_adversary,
        constit_input,
        constit_input_adversary,
        jet_input,
        jet_input_adversary,
        model_to_do,
        track_input,
        track_input_adversary,
    ) = initialize_model(training_parameters)

    # Train model
    roc_scores, dir_name = train_llp(
        training_parameters,
        model_to_do,
        constit_input,
        track_input,
        MSeg_input,
        jet_input,
        constit_input_adversary,
        track_input_adversary,
        MSeg_input_adversary,
        jet_input_adversary,
    )

    # Summarize performance metrics``
    print("Estimated AUC %.3f (%.3f)" % (np.mean(roc_scores), np.std(roc_scores)))
    logging.info(
        "Estimated AUC %.3f (%.3f)" % (np.mean(roc_scores), np.std(roc_scores))
    )
    # roc_results.append(roc_scores)
    # model_files.append(dir_name)

    # # Free up some memory
    # gc.collect()

    # # Make box plots of kFold CV results
    # if args.doKFold:
    #     process_kfold_run(
    #         roc_results, acc_results, model_to_do_list, model_files, name_list, seed
    #     )

    # # Put all model files in the same directory and save results to .txt file
    # if args.doGridSearch:
    #     process_grid_search_run(
    #         roc_results,
    #         acc_results,
    #         model_files,
    #         lr_values,
    #         reg_values,
    #         filters_cnn_constit,
    #         filters_cnn_track,
    #         filters_cnn_MSeg,
    #     )


def initialize_model(
    training_params: TrainingConfig,
) -> Tuple[
    ModelInput,
    ModelInput,
    ModelInput,
    ModelInput,
    JetInput,
    JetInput,
    str,
    ModelInput,
    ModelInput,
]:
    p_list = [
        "learning_rate",
        "filters_cnn_constit",
        "frac_list",
        "nodes_constit_lstm",
        "reg_values",
        "dropout_array",
        "adversary_weight",
    ]
    p_text = ",".join([f"{p}={getattr(training_params, p)}" for p in p_list])
    model_name = f"{training_params.model_name},{p_text}"

    logging.info(f"Initializing model {model_name}")

    assert training_params.filters_cnn_constit is not None
    assert training_params.nodes_constit_lstm is not None
    assert training_params.layers_list is not None
    assert training_params.mH_parametrization is not None
    assert training_params.mS_parametrization is not None
    assert training_params.filters_cnn_track is not None
    assert training_params.nodes_track_lstm is not None
    assert training_params.filters_cnn_MSeg is not None
    assert training_params.nodes_MSeg_lstm is not None

    constit_input = ModelInput(
        name="constit",
        rows_max=30,
        num_features=12,
        filters_cnn=training_params.filters_cnn_constit,
        nodes_lstm=training_params.nodes_constit_lstm,
        lstm_layers=training_params.layers_list,
        mH_mS_parametrization=[
            training_params.mH_parametrization,
            training_params.mS_parametrization,
        ],
    )

    track_input = ModelInput(
        name="track",
        rows_max=20,
        num_features=10,
        filters_cnn=training_params.filters_cnn_track,
        nodes_lstm=training_params.nodes_track_lstm,
        lstm_layers=training_params.layers_list,
        mH_mS_parametrization=[
            training_params.mH_parametrization,
            training_params.mS_parametrization,
        ],
    )

    MSeg_input = ModelInput(
        name="MSeg",
        rows_max=30,
        num_features=6,
        filters_cnn=training_params.filters_cnn_MSeg,
        nodes_lstm=training_params.nodes_MSeg_lstm,
        lstm_layers=training_params.layers_list,
        mH_mS_parametrization=[
            training_params.mH_parametrization,
            training_params.mS_parametrization,
        ],
    )

    jet_input = JetInput(
        name="jet",
        num_features=3,
        mH_mS_parametrization=[
            training_params.mH_parametrization,
            training_params.mS_parametrization,
        ],
    )

    # Initialize adversary input objects
    constit_input_adversary = ModelInput(
        name="constit",
        rows_max=30,
        num_features=12,
        filters_cnn=training_params.filters_cnn_constit,
        nodes_lstm=training_params.nodes_constit_lstm,
        lstm_layers=training_params.layers_list,
        mH_mS_parametrization=[
            training_params.mH_parametrization,
            training_params.mS_parametrization,
        ],
    )

    track_input_adversary = ModelInput(
        name="track",
        rows_max=20,
        num_features=10,
        filters_cnn=training_params.filters_cnn_track,
        nodes_lstm=training_params.nodes_track_lstm,
        lstm_layers=training_params.layers_list,
        mH_mS_parametrization=[
            training_params.mH_parametrization,
            training_params.mS_parametrization,
        ],
    )

    MSeg_input_adversary = ModelInput(
        name="MSeg",
        rows_max=30,
        num_features=6,
        filters_cnn=training_params.filters_cnn_MSeg,
        nodes_lstm=training_params.nodes_MSeg_lstm,
        lstm_layers=training_params.layers_list,
        mH_mS_parametrization=[
            training_params.mH_parametrization,
            training_params.mS_parametrization,
        ],
    )

    jet_input_adversary = JetInput(
        name="jet",
        num_features=3,
        mH_mS_parametrization=[
            training_params.mH_parametrization,
            training_params.mS_parametrization,
        ],
    )

    return (
        MSeg_input,
        MSeg_input_adversary,
        constit_input,
        constit_input_adversary,
        jet_input,
        jet_input_adversary,
        model_name,
        track_input,
        track_input_adversary,
    )
