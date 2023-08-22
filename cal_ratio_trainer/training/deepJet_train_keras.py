import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd
import tensorflow as tf
from keras import metrics
from keras.optimizers import Nadam
from sklearn.model_selection import train_test_split

from cal_ratio_trainer.config import TrainingConfig
from cal_ratio_trainer.training.evaluate_training import (
    checkpoint_pred_hist_main,
    do_checkpoint_prediction_histogram,
    evaluate_model,
    print_history_plots,
)
from cal_ratio_trainer.training.model_input.jet_input import JetInput
from cal_ratio_trainer.training.model_input.model_input import ModelInput
from cal_ratio_trainer.training.training_utils import (
    evaluationObject,
    prep_input_for_keras,
    prepare_training_datasets,
    setup_adversary_arrays,
    setup_model_architecture,
)
from cal_ratio_trainer.training.utils import (
    create_directories,
    load_dataset,
    low_or_high_pt_selection_train,
    match_adversary_weights,
)


def train_llp(
    training_params: TrainingConfig,
    model_to_do: str,
    constit_input: ModelInput,
    track_input: ModelInput,
    MSeg_input: ModelInput,
    jet_input: JetInput,
    constit_input_adversary: ModelInput,
    track_input_adversary: ModelInput,
    MSeg_input_adversary: ModelInput,
    jet_input_adversary: JetInput,
    plt_model: bool = False,
) -> Tuple[float, str]:
    """
    Takes in arguments to change architecture of network, does training, then runs
        evaluate_training

    :param training_params: Class of many input parameters to training
    :param model_to_do: Name of the model
    :param useGPU2: True to use GPU2
    :param constit_input: ModelInput object for constituents
    :param track_input: ModelInput object for tracks
    :param MSeg_input: ModelInput object for muon segments
    :param jet_input: ModelInput object for jets
    :param plt_model: True to save model architecture to disk
    :param skip_training: Skip training and evaluate based on directory of network given
    """
    logging.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            logging.debug(e)

    # Fail early if we aren't going to be able to find what we need.
    assert training_params.main_file is not None
    assert (
        training_params.main_file.exists()
    ), f"Main training file ({training_params.main_file}) does not exist"

    # Setup directories for output.
    logging.debug("Setting up directories...")
    # TODO: dir_name should be path(s) - rather than multiple directories
    # which require the path to be known in other places and assumptions
    # being made.
    dir_name = create_directories(
        model_to_do, os.path.split(os.path.splitext(training_params.main_file)[0])[1]
    )
    logging.debug(f"Main directory for output: {dir_name}")

    # Write a file with some details of architecture, will append final stats at end
    # of
    # training
    logging.debug("Writing to file training details...")
    training_details_file = Path("plots/") / dir_name / "training_details.txt"
    with training_details_file.open("wt+") as f_out:
        print(f"Starting Training {datetime.now()}", file=f_out)
        print(f"ModeL: {model_to_do}", file=f_out)
        print(str(training_params), file=f_out)

    # Load dataset
    logging.info(f"Loading control region data file {training_params.cr_file}...")
    df_adversary = load_dataset(training_params.cr_file)
    assert training_params.frac_list is not None
    df_adversary = match_adversary_weights(df_adversary)
    # TODO: random seed isn't passed in here to `sample`, so it will
    # be different each run.
    df_adversary = df_adversary.sample(frac=training_params.frac_list)

    logging.info(f"Loading up dataset {training_params.main_file}...")
    df = load_dataset(training_params.main_file)
    df = df.sample(frac=training_params.frac_list)

    # Extract labels
    # TODO: I do not trust how all the different weights come back, in particular,
    # how they used to be entangled due to various weird views vs copies of the
    # dataframes.
    (
        X,
        X_adversary,
        Y,
        Y_adversary,
        mcWeights,
        mcWeights_adversary,
        sig_weight,
        weights,
        weights_adversary,
        Z,
        Z_adversary,
    ) = prepare_training_datasets(df, df_adversary)

    # Split data into train/test datasets
    random_state = 1
    (
        X_train,
        X_test,
        y_train,
        y_test,
        weights_train,
        weights_test,
        mcWeights_train,
        mcWeights_test,
        Z_train,
        Z_test,
    ) = train_test_split(
        X,
        Y,
        weights,
        mcWeights,
        Z,
        test_size=0.1,
        random_state=random_state,
        shuffle=False,
    )
    (
        X_train_adversary,
        X_test_adversary,
        y_train_adversary,
        y_test_adversary,
        weights_train_adversary,
        weights_test_adversary,
        mcWeights_train_adversary,
        mcWeights_test_adversary,
        Z_train_adversary,
        Z_test_adversary,
    ) = train_test_split(
        X_adversary,
        Y_adversary,
        weights_adversary,
        mcWeights_adversary,
        Z_adversary,
        test_size=0.1,
        random_state=random_state,
        shuffle=False,
    )

    eval_object = evaluationObject()
    eval_object.fillObject_params(training_params)

    # Call method that prepares data, builds model architecture, trains model,
    # and evaluates model
    roc_auc = build_train_evaluate_model(
        constit_input,
        track_input,
        MSeg_input,
        jet_input,
        X_train,
        X_test,
        y_train,
        y_test,
        mcWeights_train,
        mcWeights_test,
        weights_train,
        weights_test,
        Z_test,
        Z_train,
        constit_input_adversary,
        track_input_adversary,
        MSeg_input_adversary,
        jet_input_adversary,
        X_train_adversary,
        X_test_adversary,
        y_train_adversary,
        y_test_adversary,
        mcWeights_train_adversary,
        mcWeights_test_adversary,
        weights_train_adversary,
        weights_test_adversary,
        Z_test_adversary,
        Z_train_adversary,
        plt_model,
        dir_name,
        eval_object,
        training_params,
    )

    #     f.close()
    #     return roc_auc, dir_name

    # else:
    #     # initialize lists to store metrics
    #     roc_scores, acc_scores = list(), list()
    #     # initialize counter for current fold iteration
    #     n_folds = 0
    #     # do KFold Cross Validation
    #     eval_object_list = []
    #     for train_ix, test_ix, train_adv_ix, test_adv_ix in (
    #         kfold.split(X, Y),
    #         kfold.split(X_adversary, Y_adversary),
    #     ):
    #         eval_object = evaluationObject()
    #         eval_object.fillObject_params(
    #             training_params.frac_list,
    #             training_params.batch_size,
    #             training_params.reg_values,
    #             training_params.dropout_array,
    #             training_params.epochs,
    #             training_params.lr_values,
    #             training_params.hidden_layer_fraction,
    #             int(constit_input.filters_cnn[3] / 4),
    #             sig_weight,
    #             int(constit_input.nodes_lstm / 40),
    #         )
    #         n_folds += 1
    #         print("\nDoing KFold iteration # %.0f...\n" % n_folds)
    #         # select samples
    #         X_test, y_test, weights_test, mcWeights_test, Z_test = (
    #             X.iloc[test_ix],
    #             Y.iloc[test_ix],
    #             weights.iloc[test_ix],
    #             mcWeights.iloc[test_ix],
    #             Z.iloc[test_ix],
    #         )
    #         X_train, y_train, weights_train, mcWeights_train, Z_train = (
    #             X.iloc[train_ix],
    #             Y.iloc[train_ix],
    #             weights.iloc[train_ix],
    #             mcWeights.iloc[train_ix],
    #             Z.iloc[train_ix],
    #         )

    #         (
    #             X_test_adversary,
    #             y_test_adversary,
    #             weights_test_adversary,
    #             mcWeights_test_adversary,
    #             Z_test_adversary,
    #         ) = (
    #             X_adversary.iloc[test_adv_ix],
    #             Y_adversary.iloc[test_adv_ix],
    #             weights_adversary.iloc[test_adv_ix],
    #             mcWeights_adversary.iloc[test_adv_ix],
    #             Z_adversary.iloc[test_adv_ix],
    #         )
    #         (
    #             X_train_adversary,
    #             y_train_adversary,
    #             weights_train_adversary,
    #             mcWeights_train_adversary,
    #             Z_train_adversary,
    #         ) = (
    #             X_adversary.iloc[train_adv_ix],
    #             Y_adversary.iloc[train_adv_ix],
    #             weights_adversary.iloc[train_adv_ix],
    #             mcWeights_adversary.iloc[train_adv_ix],
    #             Z_adversary.iloc[train_adv_ix],
    #         )

    #         # Call method that prepares data, builds model architecture, trains model,
    #         # and evaluates model
    #         roc_auc = build_train_evaluate_model(
    #             constit_input,
    #             track_input,
    #             MSeg_input,
    #             jet_input,
    #             X_train,
    #             X_test,
    #             y_train,
    #             y_test,
    #             mcWeights_train,
    #             mcWeights_test,
    #             weights_train,
    #             weights_test,
    #             Z_test,
    #             Z_train,
    #             constit_input_adversary,
    #             track_input_adversary,
    #             MSeg_input_adversary,
    #             jet_input_adversary,
    #             X_train_adversary,
    #             X_test_adversary,
    #             y_train_adversary,
    #             y_test_adversary,
    #             mcWeights_train_adversary,
    #             mcWeights_test_adversary,
    #             weights_train_adversary,
    #             weights_test_adversary,
    #             plt_model,
    #             dir_name,
    #             eval_object,
    #             useGPU2,
    #             skipTraining,
    #             training_params,
    #             kfold,
    #             n_folds,
    #         )

    #         roc_scores.append(roc_auc)
    #         eval_object_list.append(eval_object)
    #         gc.collect()
    #     evaluate_objectList(eval_object_list, f)
    #     f.close()
    #     return roc_scores, dir_name

    return roc_auc, dir_name


def _enable_layers(model, name_contains: str, does_contain: bool):
    """Enable/Disable training depending on the layer name"""
    # TODO: Move this off to a utility file somewhere. Or
    # earlier in this file.
    for _, layer_adv in enumerate(model.layers):
        if does_contain:
            layer_adv.trainable = name_contains in layer_adv.name
        else:
            layer_adv.trainable = name_contains not in layer_adv.name


def build_train_evaluate_model(
    constit_input: ModelInput,
    track_input: ModelInput,
    MSeg_input: ModelInput,
    jet_input: JetInput,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_df: pd.DataFrame,
    y_test_df: pd.DataFrame,
    mcWeights_train: pd.Series,
    mcWeights_test: pd.Series,
    weights_train: pd.Series,
    weights_test: pd.Series,
    Z_test: pd.DataFrame,
    Z_train: pd.DataFrame,
    constit_input_adversary: ModelInput,
    track_input_adversary: ModelInput,
    MSeg_input_adversary: ModelInput,
    jet_input_adversary: JetInput,
    X_train_adversary: pd.DataFrame,
    X_test_adversary: pd.DataFrame,
    y_train_adversary: pd.DataFrame,
    y_test_adversary_df: pd.DataFrame,
    mcWeights_train_adversary: pd.Series,
    mcWeights_test_adversary: pd.Series,
    weights_train_adversary: pd.Series,
    weights_test_adversary: pd.Series,
    Z_test_adversary: pd.DataFrame,
    Z_train_adversary: pd.DataFrame,
    plt_model: bool,
    dir_name: str,
    eval_object: evaluationObject,
    training_params: TrainingConfig,
):
    """
    This method has the following steps:
        - Prepares train, test, and validate data
        - Builds model architecture
        - Does model training
        - Does model evaluation
    :return: ROC area under curve metric, and model accuracy metric

    TODO: Note that we have the model, and then the adversary, with almost
          identical arguments. Feels like we should be able to organize
          this a bit better than a long list of arguments like this!
    """
    # Divide testing set into epoch-by-epoch validation and final evaluation sets
    # for the main data and the adversary.
    (
        X_test,
        X_val,
        y_test,
        y_val,
        weights_test,
        weights_val,
        mcWeights_test,
        mcWeights_val,
        Z_test,
        Z_val,
    ) = train_test_split(
        X_test,
        y_test_df,
        weights_test,
        mcWeights_test,
        Z_test,
        test_size=0.5,
    )
    (
        X_test_adversary,
        X_val_adversary,
        y_test_adversary,
        y_val_adversary,
        weights_test_adversary,
        weights_val_adversary,
        mcWeights_test_adversary,
        mcWeights_val_adversary,
        Z_test_adversary,
        Z_val_adversary,
    ) = train_test_split(
        X_test_adversary,
        y_test_adversary_df,
        weights_test_adversary,
        mcWeights_test_adversary,
        Z_test_adversary,
        test_size=0.5,
    )

    low_mass = training_params.include_low_mass
    high_mass = training_params.include_high_mass
    # TODO: these checks on not none are a mess - kill them
    # and replace with better model than we currently have.
    assert low_mass is not None
    assert high_mass is not None
    (
        X_train,
        y_train_1,
        Z_train,
        weights_train,
        mcWeights_train,
    ) = low_or_high_pt_selection_train(
        X_train,
        y_train_df,
        weights_train,
        mcWeights_train,
        Z_train,
        low_mass,
        high_mass,
    )

    X_val, y_val, Z_val, weights_val, mcWeights_val = low_or_high_pt_selection_train(
        X_val,
        y_val,
        weights_val,
        mcWeights_val,
        Z_val,
        low_mass,
        high_mass,
    )

    (
        X_test_MSeg,
        X_test_MSeg_adversary,
        X_test_constit,
        X_test_constit_adversary,
        X_test_jet,
        X_test_jet_adversary,
        X_test_track,
        X_test_track_adversary,
        X_train_MSeg,
        X_train_MSeg_adversary,
        X_train_constit,
        X_train_constit_adversary,
        X_train_jet,
        X_train_jet_adversary,
        X_train_track,
        X_train_track_adversary,
        X_val_MSeg,
        X_val_MSeg_adversary,
        X_val_constit,
        X_val_constit_adversary,
        X_val_jet,
        X_val_jet_adversary,
        X_val_track,
        X_val_track_adversary,
        y_train,
        y_val,
    ) = prep_input_for_keras(
        MSeg_input,
        X_test,
        X_test_adversary,
        X_train,
        X_train_adversary,
        X_val,
        X_val_adversary,
        Z_test,
        Z_test_adversary,
        Z_train,
        Z_train_adversary,
        Z_val,
        Z_val_adversary,
        constit_input,
        jet_input,
        track_input,
        y_train_1,
        y_val,
    )

    logging.debug("Done preparing data for model")

    # Setup training inputs, outputs, and weights
    x_to_train = [X_train_constit, X_train_track, X_train_MSeg, X_train_jet.values]
    x_to_adversary = [
        X_train_constit_adversary,
        X_train_track_adversary,
        X_train_MSeg_adversary,
        X_train_jet_adversary.values,
    ]
    y_to_train = [y_train, y_train, y_train, y_train, y_train, y_train]
    y_to_train_adversary = [y_train_adversary]
    weights_to_train = [
        weights_train.values,
        weights_train.values,
        weights_train.values,
        weights_train.values,
        weights_train.values,
        weights_train.values,
    ]

    # Setup validation inputs, outputs, and weights
    x_to_validate = [X_val_constit, X_val_track, X_val_MSeg, X_val_jet.values]
    x_to_validate_adv = [
        X_val_constit_adversary,
        X_val_track_adversary,
        X_val_MSeg_adversary,
        X_val_jet_adversary.values,
    ]
    y_to_validate = [y_val, y_val, y_val, y_val, y_val, y_val]
    y_to_validate_adv = [y_val_adversary]
    weights_to_validate = [
        weights_val.values,
        weights_val.values,
        weights_val.values,
        weights_val.values,
        weights_val.values,
        weights_val.values,
    ]

    # Setup testing input, outputs, and weights
    x_to_test = [X_test_constit, X_test_track, X_test_MSeg, X_test_jet.values]
    x_to_test_adversary = [
        X_test_constit_adversary,
        X_test_track_adversary,
        X_test_MSeg_adversary,
        X_test_jet_adversary.values,
    ]
    weights_to_test = [
        weights_test.values,
        weights_test.values,
        weights_test.values,
        weights_test.values,
        weights_test.values,
        weights_test.values,
    ]

    # Now to setup ML architecture
    logging.debug("Setting up model architecture...")
    (
        original_model,
        discriminator_model,
        discriminator_out,
        final_model,
    ) = setup_model_architecture(
        constit_input,
        track_input,
        MSeg_input,
        jet_input,
        X_train_constit,
        X_train_track,
        X_train_MSeg,
        X_train_jet,
        training_params,
    )

    # Show summary of model architecture
    original_model.save(
        "keras_outputs/" + dir_name + "/model.keras"
    )  # creates a HDF5 file
    final_model.save(
        "keras_outputs/" + dir_name + "/final_model.keras"
    )  # creates a HDF5 file
    discriminator_model.save(
        "keras_outputs/" + dir_name + "/discriminator_model.keras"
    )  # creates a HDF5 file
    discriminator_model.save_weights(
        "keras_outputs/" + dir_name + "/initial_discriminator_model_weights.keras"
    )

    # Do training
    logging.info("Starting training")
    (
        accept_epoch_array,
        adv_acc,
        adv_loss,
        advw_array,
        checkpoint_ks_bib,
        checkpoint_ks_qcd,
        checkpoint_ks_sig,
        epoch_list,
        ks_bib_hist,
        ks_qcd_hist,
        ks_sig_hist,
        lr_array,
        num_epochs,
        num_splits,
        original_acc,
        original_adv_acc,
        original_adv_lossf,
        original_lossf,
        small_mcWeights_val_adversary,
        small_weights_train_adversary,
        small_weights_val_adversary,
        small_x_to_adversary_split,
        small_x_val_adversary,
        small_y_to_train_adversary_0,
        small_y_val_adversary,
        stable_counter,
        val_adv_acc,
        val_adv_loss,
        val_original_acc,
        val_original_adv_acc,
        val_original_adv_lossf,
        val_original_lossf,
        weights_to_train_0,
        weights_to_validate_0,
        weights_train_adversary_s,
        weights_val_adversary_split,
        x_to_adversary_split,
        x_to_train_split,
        x_to_validate_adv_split,
        x_to_validate_split,
        y_to_train_0,
        y_to_train_adversary_squeeze,
        y_to_validate_0,
        y_to_validate_adv_squeeze,
    ) = setup_adversary_arrays(
        mcWeights_val_adversary,
        weights_to_train,  # type: ignore
        weights_to_validate,  # type: ignore
        weights_train_adversary,
        weights_val_adversary,
        x_to_adversary,
        x_to_train,
        x_to_validate,
        x_to_validate_adv,
        y_to_train,
        y_to_train_adversary,  # type: ignore
        y_to_validate,
        y_to_validate_adv,
        training_params,
    )

    # Summarize the two models
    logging.debug("Summary of the original model:")
    _enable_layers(original_model, "adversary", False)
    original_model.summary(print_fn=logging.debug)
    logging.debug("Summary of the discriminator model:")
    _enable_layers(discriminator_model, "adversary", True)
    discriminator_model.summary(print_fn=logging.debug)

    # Create the optimizers for the main training and the adversary training.
    assert training_params.lr_values is not None
    optimizer = Nadam(
        learning_rate=training_params.lr_values,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
    )
    optimizer_adv = Nadam(
        learning_rate=training_params.lr_values,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
    )

    # Train each epoch
    assert training_params.adversary_weight is not None
    for i_epoch in epoch_list:
        logging.info(f"Training Epoch {i_epoch+1} of {len(epoch_list)}")
        # Set up decaying learning rate
        current_lr = training_params.lr_values * (1.0 / (1.0 + 0.03 * i_epoch))
        # Set up increasing adversary weight, increases per epoch
        current_adversary_weight = training_params.adversary_weight * (
            1.0 / (1.0 - 0.005 * i_epoch)
        )
        main_model_weight = 1 * (1.0 / (1.0 + 0.01125 * i_epoch))
        logging.debug(f"main model weight: {main_model_weight}")
        logging.debug(f"adv weight: {current_adversary_weight}")

        last_loss = -1
        last_main_output_loss = -1
        last_adversary_loss = -1
        last_main_cat_acc = -1
        last_adv_bin_acc = -1

        last_disc_loss = -1
        last_disc_bin_acc = -1

        # Training: the run main and adversary once each for every
        # mini-batch

        for i_batch in range(num_splits):
            logging.debug(
                f"Running batch {i_batch+1} (of {num_splits} batches) of "
                f"epoch {i_epoch+1}"
            )

            train_inputs = [*x_to_train_split[i_batch], *x_to_adversary_split[i_batch]]
            train_outputs = [
                y_to_train_0[i_batch],
                y_to_train_adversary_squeeze[i_batch],
            ]
            train_weights = [
                weights_to_train_0[i_batch],
                weights_train_adversary_s[i_batch],
            ]

            # Train the adversary network
            logging.debug("  -> Training adversary")

            # Labeling as adversary in order to only train adv layers
            _enable_layers(discriminator_model, "adversary", True)

            # training adversary twice with two different learning rates
            for x in [19, 0.1]:
                optimizer_adv.learning_rate.assign(current_lr * x)
                logging.debug(
                    "  Training adversary with "
                    f"lr {optimizer_adv.learning_rate.value()}"
                )

                discriminator_model.compile(
                    optimizer=optimizer_adv,
                    loss="binary_crossentropy",
                    metrics=[metrics.binary_accuracy],
                )
                adversary_hist = discriminator_model.train_on_batch(
                    small_x_to_adversary_split[i_batch],
                    small_y_to_train_adversary_0[i_batch],
                    sample_weight=small_weights_train_adversary[i_batch],
                )

                last_disc_loss = adversary_hist[0]
                last_disc_bin_acc = adversary_hist[1]
                logging.debug(f"  Adversary Loss: {last_disc_loss:.4f}")
                logging.debug(f"  Adversary binary Accuracy: {last_disc_bin_acc:.4f}")

            logging.debug("  -> Training main network")

            _enable_layers(original_model, "adversary", False)

            # TODO: This is the second place we are setting the
            # learning rate - do we need it in both places?
            # Is there any point in using Adam if we are re-creating it
            # for every single iteration?
            optimizer.learning_rate.assign(current_lr)

            original_model.compile(
                optimizer=optimizer,
                loss=["categorical_crossentropy", "binary_crossentropy"],
                metrics=[metrics.categorical_accuracy, metrics.binary_accuracy],
                loss_weights=[1 * main_model_weight, -current_adversary_weight],
            )

            original_hist = original_model.train_on_batch(
                train_inputs, train_outputs, train_weights
            )

            last_loss = original_hist[0]
            last_main_output_loss = original_hist[1]
            last_adversary_loss = original_hist[2]
            last_main_cat_acc = original_hist[3]
            last_adv_bin_acc = original_hist[6]

        logging.info(f"  loss: {last_loss:.4f}")
        logging.info(f"  main_output_loss: {last_main_output_loss:.4f}")
        logging.info(f"  adversary_loss: {last_adversary_loss:.4f}")
        logging.debug(f"  Main categorical accuracy: {last_main_cat_acc}")
        logging.debug(f"  Adversary binary accuracy: {last_adv_bin_acc}")

        # At the end of the epoch run testing.
        logging.debug("End of Epoch Testing")

        # Do test on small batch
        original_val_hist = original_model.test_on_batch(
            [*(x_to_validate_split[0]), *(x_to_validate_adv_split[0])],
            [y_to_validate_0[0], y_to_validate_adv_squeeze[0]],
            [weights_to_validate_0[0], weights_val_adversary_split[0]],
        )
        val_last_loss = original_val_hist[0]
        val_last_main_output_loss = original_val_hist[1]
        val_last_adversary_loss = original_val_hist[2]
        val_last_main_cat_acc = original_val_hist[3]
        val_last_adv_bin_acc = original_val_hist[6]
        logging.debug(f"val loss: {val_last_loss:.4f}")
        logging.debug(f"val main_output_loss: {val_last_main_output_loss:.4f}")
        logging.debug(f"val adversary_loss: {val_last_adversary_loss:.4f}")
        logging.debug(f"val Main categorical accuracy: {val_last_main_cat_acc}")
        logging.debug(f"val Adversary binary accuracy: {val_last_adv_bin_acc}")

        adversary_val_hist = discriminator_model.test_on_batch(
            small_x_val_adversary,
            small_y_val_adversary[0],
            small_weights_val_adversary,
        )
        val_last_disc_loss = adversary_val_hist[0]
        val_last_disc_bin_acc = adversary_val_hist[1]
        logging.debug(f"Val Adversary Loss: {val_last_disc_loss:.4f}")
        logging.debug(f"Val Adversary binary Accuracy: {val_last_disc_bin_acc:.4f}")

        # Check to see if things have gotten better
        ks_qcd, ks_sig, ks_bib = do_checkpoint_prediction_histogram(
            final_model,
            dir_name,
            small_x_val_adversary,
            small_y_val_adversary[0],
            small_mcWeights_val_adversary,
            str(i_epoch) + "_val",
            high_mass,
            low_mass,
        )

        # Every epoch save weights if KS test below some threshold (0.3 seems good)
        if ks_bib < 0.3:
            final_model.save_weights(
                "keras_outputs/" + dir_name + f"/final_model_weights_{i_epoch}.keras"
            )

        final_model.save_weights(
            "keras_outputs/" + dir_name + "/final_model_weights.keras"
        )
        original_model.save_weights("keras_outputs/" + dir_name + "/checkpoint.keras")
        discriminator_model.save_weights(
            "keras_outputs/" + dir_name + "/adv_checkpoint.keras"
        )
        #         print("Clear session")
        #         K.clear_session()
        #         print("Reload models and weights")
        #         (
        #             original_model,
        #             discriminator_model,
        #             discriminator_out,
        #             final_model,
        #         ) = setup_model_architecture(
        #             constit_input,
        #             track_input,
        #             MSeg_input,
        #             jet_input,
        #             X_train_constit,
        #             X_train_track,
        #             X_train_MSeg,
        #             X_train_jet,
        #             x_to_adversary,
        #             y_to_train_adversary,
        #             weights_train_adversary_s,
        #             training_params,
        #         )
        #         stable_counter += 1

        #         accept_epoch = False
        #         final_model.load_weights(
        #             "keras_outputs/" + dir_name + f"/final_model_weights.h5"
        #         )
        #         original_model.load_weights("keras_outputs/"
        #           + dir_name + "/checkpoint.h5")
        #         discriminator_model.load_weights(
        #             "keras_outputs/" + dir_name + "/adv_checkpoint.h5"
        #         )

        #         print("Saving model weights")
        #         final_model.save_weights(
        #             "keras_outputs/" + dir_name + "/previous_final_model_weights.h5"
        #         )
        #         original_model.save_weights(
        #             "keras_outputs/" + dir_name + "/previous_checkpoint.h5"
        #         )
        #         discriminator_model.save_weights(
        #             "keras_outputs/" + dir_name + "/previous_adv_checkpoint.h5"
        #         )

        #         accept_epoch_array.append(int(accept_epoch))

        ks_qcd_hist.append(ks_qcd)
        ks_sig_hist.append(ks_sig)
        ks_bib_hist.append(ks_bib)

        # Adding the train and test loss to a text function to save the loss
        # Helpful to monitor training performance with large variations in loss
        # making the graph hard to parse

        with open("plots/" + dir_name + "train_loss.txt", "w") as train_file:
            train_file.write(str(last_main_output_loss) + "\n")
        with open("plots/" + dir_name + "test_loss.txt", "w") as test_file:
            test_file.write(str(val_last_main_output_loss) + "\n")

        # generating checkpoint plots of the model performance during training
        checkpoint_pred_hist_main(
            final_model,
            dir_name,
            x_to_test,
            y_test,
            mcWeights_test,
            i_epoch,
            high_mass,
            low_mass,
        )

        # Append some lists with stats of latest epoch
        # and dump them out so they can be seen in "real-time"
        advw_array.append(current_adversary_weight)
        lr_array.append(current_lr)
        adv_loss.append(last_disc_loss)
        adv_acc.append(last_disc_bin_acc)
        val_adv_loss.append(val_last_disc_loss)
        val_adv_acc.append(val_last_disc_bin_acc)
        original_lossf.append(last_main_output_loss)
        original_acc.append(last_main_cat_acc)
        val_original_lossf.append(val_last_main_output_loss)
        val_original_acc.append(val_last_main_cat_acc)
        original_adv_lossf.append(last_adversary_loss)
        original_adv_acc.append(last_adv_bin_acc)
        val_original_adv_lossf.append(val_last_adversary_loss)
        val_original_adv_acc.append(val_last_adv_bin_acc)

        print_history_plots(
            advw_array,
            adv_loss,
            adv_acc,
            val_adv_loss,
            val_adv_acc,
            original_lossf,
            original_acc,
            val_original_lossf,
            val_original_acc,
            original_adv_lossf,
            original_adv_acc,
            val_original_adv_lossf,
            val_original_adv_acc,
            lr_array,
            ks_qcd_hist,
            ks_sig_hist,
            ks_bib_hist,
            accept_epoch_array,
            dir_name,
        )
    logging.info("Finished training")

    #     # Save model weights
    #     original_model.save_weights("keras_outputs/" + dir_name + "/model_weights.h5")
    #     final_model.save_weights(
    #         "keras_outputs/" + dir_name + "/final_model_weights.h5"
    #     )
    #     discriminator_model.save_weights(
    #         "keras_outputs/" + dir_name + "/discriminator_model_weights.h5"
    #     )
    #     del original_model  # deletes the existing model

    #     # initialize model with same architecture
    #     model = load_model("keras_outputs/" + dir_name + "/final_model.h5")
    #     discriminator_model = load_model(
    #         "keras_outputs/" + dir_name + "/discriminator_model.h5",
    #         custom_objects={"DenseSN": DenseSN},
    #     )
    #     # load weights
    #     model.load_weights("keras_outputs/" + dir_name + "/final_model_weights.h5")
    #     discriminator_model.load_weights(
    #         "keras_outputs/" + dir_name + "/adv_checkpoint.h5"
    #     )

    # Evaluate Model with ROC curves
    logging.debug("Evaluating model...")
    # TODO: improve doc on Z and mcWeights
    roc_auc, SoverB = evaluate_model(
        final_model,
        discriminator_model,
        dir_name,
        x_to_test,
        y_test,
        weights_to_test,  # type: ignore
        Z_test,
        mcWeights_test,
        x_to_test_adversary,
        y_test_adversary,
        weights_test_adversary,
        None,
        eval_object,
        Z_test_adversary,
        high_mass,
        low_mass,
    )
    logging.info("ROC area under curve: %.3f" % roc_auc)
    logging.info("Max S over Root B: %.3f" % SoverB)

    #     return roc_auc
    # # This happens if we skip training
    # else:
    #     # initialize model with same architecture
    #     # model = load_model('keras_outputs/' + skipTraining[1] + '/final_model.h5')
    #     (
    #         original_model,
    #         discriminator_model,
    #         discriminator_out,
    #         model,
    #     ) = setup_model_architecture(
    #         constit_input,
    #         track_input,
    #         MSeg_input,
    #         jet_input,
    #         X_train_constit,
    #         X_train_track,
    #         X_train_MSeg,
    #         X_train_jet,
    #         x_to_adversary,
    #         y_to_train_adversary,
    #         weights_train_adversary_s,
    #         training_params,
    #     )
    #     # model = load_model('keras_outputs/' + skipTraining[1]
    #     #      + '/cpu_model_weights_hm_0_resume.h5')
    #     # load weights
    #     model_architecture = (
    #         "keras_outputs/"
    #         + skipTraining[1]
    #         + "/final_keras_cpu_model_hm_0_apr28.json"
    #     )
    #     with open(model_architecture, "r") as json_file:
    #         # architecture = json.load(json_file)
    #         model = model_from_json(json_file.read())
    #     # model = tf.compat.v1.keras.models.model_from_json('keras_outputs/'
    #     #      + skipTraining[1] + '/final_keras_cpu_model_hm_0_apr28.json')
    #     model.load_weights(
    #         "keras_outputs/" + skipTraining[1] + "/cpu_model_weights_hm_0_resume.h5"
    #     )
    #     optimizer = Nadam(
    #         learning_rate=current_lr,
    #         beta_1=0.9,
    #         beta_2=0.999,
    #         epsilon=1e-07,
    #         schedule_decay=0.05,
    #     )
    #     model.compile(
    #         optimizer=optimizer,
    #         loss="categorical_crossentropy",
    #         metrics=[metrics.categorical_accuracy],
    #     )
    #     # model.save('keras_outputs/backups/cpu_model_hm_apr12.h5',
    #           include_optimizer=False)  # creates a HDF5 file
    #     # model.save('keras_outputs/f deep/final_keras_cpu_model_lm_1_apr28.h5')

    #     # Evaluate Model with ROC curves
    #     print("\nEvaluating model...\n")
    #     # TODO: improve doc on Z and mcWeights
    #     roc_auc, SoverB = evaluate_model(
    #         model,
    #         discriminator_model,
    #         dir_name,
    #         x_to_test,
    #         y_test,
    #         weights_to_test,
    #         Z_test,
    #         mcWeights_test,
    #         x_to_test_adversary,
    #         y_test_adversary,
    #         weights_test_adversary,
    #         n_folds,
    #         eval_object,
    #         Z_test_adversary,
    #         skipTraining[0],
    #     )
    #     print("ROC area under curve: %.3f" % roc_auc)
    #     print("Max S over Root B: %.3f" % SoverB)

    return roc_auc
