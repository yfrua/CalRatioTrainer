import logging
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from keras import metrics
from keras.layers import BatchNormalization, Dense, Dropout, concatenate
from keras.models import Model
from keras.optimizers import Nadam
from keras.regularizers import L1L2
# from keras.src.utils import np_utils
from keras.utils import to_categorical

from tensorflow import Tensor

from cal_ratio_trainer.config import TrainingConfig
from cal_ratio_trainer.training.model_input.jet_input import JetInput
from cal_ratio_trainer.training.model_input.model_input import ModelInput
from cal_ratio_trainer.common.column_names import (
    col_cluster_track_mseg_names,
    col_jet_names,
    col_llp_mass_names,
)


def prepare_training_datasets(
    df: pd.DataFrame, df_adversary: pd.DataFrame
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    np.ndarray,
    pd.Series,
    pd.Series,
    float,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Sets up dataframes into X (input variables), Y (labels), weights (jet weights),
    Z (Extra info)

    :param df: main jet dataframe
    :param df_adversary: CR jet dataframe
    :return: X,Y,Z, main jets and adversary
    """
    Y = df["label"]
    Y_adversary = df_adversary["label"].copy()
    Y_adversary.loc[Y_adversary == 2] = 1
    Y_adversary = np.array(Y_adversary.values)

    # Pull out the weights for later use
    weights = df["mcEventWeight"]
    weights_adversary = df_adversary["mcEventWeight"]
    mcWeights = df["mcEventWeight"].copy()
    mcWeights_adversary = df_adversary["mcEventWeight"]

    # Rescale the weights so that signal and qcd have the same weight.
    # TODO: Adversary rescaling is done in utils.py, in match_adversary_weights.
    #       Make this symmetric so re-scaling is done in the "same" place.
    qcd_weight = cast(float, np.sum(mcWeights[Y == 0]))
    sig_weight = cast(float, np.sum(mcWeights[Y == 1]))

    logging.debug("mcWeights")
    logging.debug("label 0: " + str(mcWeights[Y == 0]))

    mcWeights.loc[Y == 0] = mcWeights[Y == 0] * (sig_weight / qcd_weight)

    # TODO: Why not rescale BIB so it also has the same weight?
    # Perhaps because it is data?

    # Build the arrays from an explicit list of columns
    X = df.loc[:, col_cluster_track_mseg_names + col_jet_names + ["eventNumber"]]
    X_adversary = df_adversary.loc[
        :, col_cluster_track_mseg_names + col_jet_names + ["eventNumber"]
    ]
    Z = df.loc[:, col_llp_mass_names]
    Z_adversary = df_adversary.loc[:, ["jet_pt", "jet_eta"]]

    # Pack it up and return it all!
    return (
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
    )


class evaluationObject:
    def __init__(self):
        self.training_params: Optional[TrainingConfig] = None

        # Values we track
        self.all_significance = -1
        self.all_auc = -1
        self.all_aus = -1
        self.qcd_significance = -1
        self.qcd_aus = -1
        self.bib_significance = -1
        self.bib_aus = -1
        self.mh60_significance = -1
        self.mh60_auc = -1
        self.mh60_aus = -1
        self.mh125_significance = -1
        self.mh125_auc = -1
        self.mh125_aus = -1
        self.mh200_significance = -1
        self.mh200_auc = -1
        self.mh200_aus = -1
        self.mh400_significance = -1
        self.mh400_auc = -1
        self.mh400_aus = -1
        self.mh600_significance = -1
        self.mh600_auc = -1
        self.mh600_aus = -1
        self.mh1000_significance = -1
        self.mh1000_auc = -1
        self.mh1000_aus = -1

    def fillObject_sOverB(self, label, value):
        if "600" in label:
            self.mh600_significance = value
        elif "60" in label:
            self.mh60_significance = value
        elif "125" in label:
            self.mh125_significance = value
        elif "200" in label:
            self.mh200_significance = value
        elif "400" in label:
            self.mh400_significance = value
        elif "1000" in label:
            self.mh1000_significance = value
        elif "BIB" in label:
            self.bib_significance = value
        elif "QCD" in label:
            self.qcd_significance = value
        else:
            self.all_significance = value

    def fillObject_auc(self, label, value):
        if "600" in label:
            self.mh600_auc = value
        elif "60" in label:
            self.mh60_auc = value
        elif "125" in label:
            self.mh125_auc = value
        elif "200" in label:
            self.mh200_auc = value
        elif "400" in label:
            self.mh400_auc = value
        elif "600" in label:
            self.mh600_auc = value
        elif "1000" in label:
            self.mh1000_auc = value
        else:
            self.all_auc = value

    def fillObject_aus(self, label, value):
        if "600" in label:
            self.mh600_aus = value
        elif "60" in label:
            self.mh60_aus = value
        elif "125" in label:
            self.mh125_aus = value
        elif "200" in label:
            self.mh200_aus = value
        elif "400" in label:
            self.mh400_aus = value
        elif "600" in label:
            self.mh600_aus = value
        elif "1000" in label:
            self.mh1000_aus = value
        elif "BIB" in label:
            self.bib_aus = value
        elif "QCD" in label:
            self.qcd_aus = value
        else:
            self.all_aus = value

    def fillObject_params(self, training_params: TrainingConfig):
        self.training_params = training_params


def prep_input_for_keras(
    MSeg_input: ModelInput,
    X_test: pd.DataFrame,
    X_test_adversary: pd.DataFrame,
    X_train: pd.DataFrame,
    X_train_adversary: pd.DataFrame,
    X_val: pd.DataFrame,
    X_val_adversary: pd.DataFrame,
    Z_test: pd.DataFrame,
    Z_test_adversary: pd.DataFrame,
    Z_train: pd.DataFrame,
    Z_train_adversary: pd.DataFrame,
    Z_val: pd.DataFrame,
    Z_val_adversary: pd.DataFrame,
    constit_input: ModelInput,
    jet_input: JetInput,
    track_input: ModelInput,
    y_train_df: pd.DataFrame,
    y_val_df: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Need inputs to be in very specific shapes for Conv1D+LSTM training

    :param MSeg_input: muon segment format
    :param X_test: testing set input, main jets
    :param X_test_adversary: testing set input, CR jets
    :param X_train: training set input, main jets
    :param X_train_adversary: training set input, CR jets
    :param X_val: validation set input, main jets
    :param X_val_adversary: validation set input, CR jets
    :param Z_test: Z for main jets, testing set
    :param Z_test_adversary: Z for CR jets, testing set
    :param Z_train: Z for main jets, training set
    :param Z_train_adversary: Z for CR jets, training set
    :param Z_val: Z for main jets, validation set
    :param Z_val_adversary: Z for CR jets, validation set
    :param constit_input: constituent/topocluster format
    :param jet_input: jet format
    :param track_input: track format
    :param y_train: training set labels
    :param y_val: validation set labels
    :return:
    """

    # Convert labels to categorical (needed for multiclass training)
    y_train = to_categorical(y_train_df)
    y_val = to_categorical(y_val_df)

    # Split X into track, MSeg, and constit inputs and reshape dataframes into
    # shape expected by Keras. This is an ordered array, so each input is
    # formatted as number of constituents x number of variables
    logging.debug("Preparing jet constituent data")
    (
        X_train_constit,
        X_val_constit,
        X_test_constit,
    ) = constit_input.extract_and_split_data(
        X_train, X_val, X_test, Z_train, Z_val, Z_test, "clus_pt_0", "clus_time_"
    )
    logging.debug("Preparing jet constituent data (adversary)")
    (
        X_train_constit_adversary,
        X_val_constit_adversary,
        X_test_constit_adversary,
    ) = constit_input.extract_and_split_data(
        X_train_adversary,
        X_val_adversary,
        X_test_adversary,
        Z_train_adversary,
        Z_val_adversary,
        Z_test_adversary,
        "clus_pt_0",
        "clus_time_",
    )

    logging.debug("Preparing track data")
    X_train_track, X_val_track, X_test_track = track_input.extract_and_split_data(
        X_train,
        X_val,
        X_test,
        Z_train,
        Z_val,
        Z_test,
        "track_pt_0",
        "track_SCTHits_",
    )
    logging.debug("Preparing track data (adversary)")
    (
        X_train_track_adversary,
        X_val_track_adversary,
        X_test_track_adversary,
    ) = track_input.extract_and_split_data(
        X_train_adversary,
        X_val_adversary,
        X_test_adversary,
        Z_train_adversary,
        Z_val_adversary,
        Z_test_adversary,
        "track_pt_0",
        "track_SCTHits_",
    )

    logging.debug("Preparing MSeg data")
    X_train_MSeg, X_val_MSeg, X_test_MSeg = MSeg_input.extract_and_split_data(
        X_train,
        X_val,
        X_test,
        Z_train,
        Z_val,
        Z_test,
        "MSeg_etaPos_0",
        "MSeg_t0_",
    )
    logging.debug("Preparing MSeg data (adversary)")
    (
        X_train_MSeg_adversary,
        X_val_MSeg_adversary,
        X_test_MSeg_adversary,
    ) = MSeg_input.extract_and_split_data(
        X_train_adversary,
        X_val_adversary,
        X_test_adversary,
        Z_train_adversary,
        Z_val_adversary,
        Z_test_adversary,
        "MSeg_etaPos_0",
        "MSeg_t0_",
    )

    logging.debug("Preparing jet data")
    X_train_jet, X_val_jet, X_test_jet = jet_input.extract_and_split_data(
        X_train, X_val, X_test, Z_train, Z_val, Z_test, "jet_pt", "jet_phi"
    )
    (
        X_train_jet_adversary,
        X_val_jet_adversary,
        X_test_jet_adversary,
    ) = jet_input.extract_and_split_data(
        X_train_adversary,
        X_val_adversary,
        X_test_adversary,
        Z_train_adversary,
        Z_val_adversary,
        Z_test_adversary,
        "jet_pt",
        "jet_phi",
    )

    return (
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
    )


def setup_model_architecture(
    constit_input: ModelInput,
    track_input: ModelInput,
    MSeg_input: ModelInput,
    jet_input: JetInput,
    X_train_constit: np.ndarray,
    X_train_track: np.ndarray,
    X_train_MSeg: np.ndarray,
    X_train_jet: pd.DataFrame,
    training_params: TrainingConfig,
) -> Tuple[Model, Model, Tensor, Model]:
    """Method that builds the model architecture and returns the model objects"""
    # Set up inputs and outputs for Keras layers
    # This sets up the layers specified in the ModelInput object i.e. Conv1D, LSTM
    # The Any above is a KerasTensor, which is a private class, and not like a Tensor.

    (
        constit_input_tensor_cr,
        constit_output_tensor_cr,
        constit_dense_tensor_cr,
        constit_input_tensor_adv,
        constit_output_tensor_adv,
        constit_dense_tensor_adv,
    ) = constit_input.init_keras_layers(X_train_constit[0].shape, training_params)

    (
        track_input_tensor_cr,
        track_output_tensor_cr,
        track_dense_tensor_cr,
        track_input_tensor_adv,
        track_output_tensor_adv,
        track_dense_tensor_adv,
    ) = track_input.init_keras_layers(X_train_track[0].shape, training_params)
    (
        MSeg_input_tensor_cr,
        MSeg_output_tensor_cr,
        MSeg_dense_tensor_cr,
        MSeg_input_tensor_adv,
        MSeg_output_tensor_adv,
        MSeg_dense_tensor_adv,
    ) = MSeg_input.init_keras_layers(X_train_MSeg[0].shape, training_params)

    # Set up layers for jet
    (
        jet_input_tensor_cr,
        jet_output_tensor_cr,
        jet_input_tensor_adv,
        jet_output_tensor_adv,
    ) = jet_input.init_keras_dense_input_output(X_train_jet.values[0].shape)
    # Setup concatenation layer
    concat_tensor_cr = concatenate(
        [
            constit_output_tensor_cr,
            track_output_tensor_cr,
            MSeg_output_tensor_cr,
            jet_input_tensor_cr,
        ]
    )
    concat_tensor_adv = concatenate(
        [
            constit_output_tensor_adv,
            track_output_tensor_adv,
            MSeg_output_tensor_adv,
            jet_input_tensor_adv,
        ]
    )
    # Setup Dense + Dropout layers

    assert training_params.hidden_layer_fraction is not None
    assert training_params.reg_values is not None
    concat_tensor = Dense(
        int(training_params.hidden_layer_fraction * 512),
        activation="relu",
        kernel_regularizer=L1L2(
            l1=training_params.reg_values, l2=training_params.reg_values
        ),
    )
    concat_tensor_cr = concat_tensor(concat_tensor_cr)
    concat_tensor_adv = concat_tensor(concat_tensor_adv)

    concat_tensor = Dropout(training_params.dropout_array)
    concat_tensor_cr = concat_tensor(concat_tensor_cr)
    concat_tensor_adv = concat_tensor(concat_tensor_adv)

    concat_tensor = Dense(
        int(training_params.hidden_layer_fraction * 64),
        activation="relu",
        kernel_regularizer=L1L2(
            l1=training_params.reg_values, l2=training_params.reg_values
        ),
    )
    concat_tensor_cr = concat_tensor(concat_tensor_cr)
    concat_tensor_adv = concat_tensor(concat_tensor_adv)

    concat_tensor = Dropout(training_params.dropout_array)
    concat_tensor_cr = concat_tensor(concat_tensor_cr)
    concat_tensor_adv = concat_tensor(concat_tensor_adv)

    # Setup final layer
    main_output_tensor = Dense(3, activation="softmax", name="main_output")
    main_output_tensor_cr = main_output_tensor(concat_tensor_cr)
    main_output_tensor_adv = main_output_tensor(concat_tensor_adv)

    # For adversary
    discriminator = BatchNormalization(name="adversary_norm_1")(main_output_tensor_adv)
    discriminator = Dense(
        24, name="adversary_1", activation="tanh", kernel_initializer="glorot_uniform"
    )(discriminator)
    discriminator = BatchNormalization(name="adversary_norm_2")(discriminator)
    discriminator = Dense(
        12, name="adversary_2", activation="tanh", kernel_initializer="glorot_uniform"
    )(discriminator)
    discriminator = BatchNormalization(name="adversary_norm_3")(discriminator)
    discriminator = Dense(
        6, name="adversary_3", activation="tanh", kernel_initializer="glorot_uniform"
    )(discriminator)
    discriminator = BatchNormalization(name="adversary_norm_4")(discriminator)

    discriminator_out_r = Dense(
        1,
        activation="sigmoid",
        name="adversary_out",
        kernel_initializer="glorot_uniform",
    )(discriminator)
    # Get around type checking
    assert discriminator_out_r is not None
    discriminator_out = discriminator_out_r  # type: Tensor

    # Setup training layers
    layers_to_input = [
        constit_input_tensor_cr,
        track_input_tensor_cr,
        MSeg_input_tensor_cr,
        jet_input_tensor_cr,
        constit_input_tensor_adv,
        track_input_tensor_adv,
        MSeg_input_tensor_adv,
        jet_input_tensor_adv,
    ]
    layers_to_output = [main_output_tensor_cr, discriminator_out]

    # Setup Model
    model = Model(
        inputs=layers_to_input, outputs=layers_to_output, name="original_model"
    )
    model_discriminator = Model(
        inputs=[
            constit_input_tensor_adv,
            track_input_tensor_adv,
            MSeg_input_tensor_adv,
            jet_input_tensor_adv,
        ],
        outputs=discriminator_out,
        name="discriminator_model",
    )
    model_final = Model(
        inputs=[
            constit_input_tensor_cr,
            track_input_tensor_cr,
            MSeg_input_tensor_cr,
            jet_input_tensor_cr,
        ],
        outputs=main_output_tensor_cr,
        name="final_model",
    )

    # Setup optimizer (N adam is good as it has decaying learning rate)
    assert training_params.lr_values is not None
    optimizer = Nadam(
        learning_rate=training_params.lr_values,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
    )

    # Compile discriminator Model
    # TODO: don't we compile inside the main loop? Is this needed?
    for layer_index_adv, layer_adv in enumerate(model_discriminator.layers):
        layer_adv.trainable = "adversary" in layer_adv.name
    model_discriminator.compile(
        optimizer="SGD",
        loss="binary_crossentropy",
        metrics=[metrics.categorical_accuracy],
    )

    for layer_index, layer in enumerate(model.layers):
        layer.trainable = "adversary" not in layer.name
    # TODO: this next line was uncommented, but perhaps for debugging?
    # n_gen_trainable = len(model.trainable_weights)

    # Compile main Model
    assert training_params.adversary_weight is not None
    model.compile(
        optimizer=optimizer,
        loss=["categorical_crossentropy", "binary_crossentropy"],
        metrics=[metrics.categorical_accuracy],
        loss_weights=[1, -training_params.adversary_weight],
    )

    # Compile final Model
    model_final = Model(
        inputs=[
            constit_input_tensor_cr,
            track_input_tensor_cr,
            MSeg_input_tensor_cr,
            jet_input_tensor_cr,
        ],
        outputs=main_output_tensor_cr,
    )
    model_final.compile(
        optimizer=optimizer,
        loss=["categorical_crossentropy"],
        metrics=[metrics.categorical_accuracy],
    )

    return model, model_discriminator, discriminator_out, model_final


def _pad_arrays(
    array_to_pad,
    array_to_match,
):
    """
    Pads arrays in `array_to_pad` to match the length of arrays in `array_to_match`.
    If the lengths of the two lists of arrays do not match, a warning is logged.

        Args:
        array_to_pad: A list of numpy arrays to be padded.
        array_to_match: A list of numpy arrays or pandas dataframes to match
            the shape of `array_to_pad`.

    Note: Types are specified on purpose here - the typing is rather
    tough because sometimes a copy is returned and other times the
    original is returned. Plus the original code has `x = _pad_arrays(x, y)`,
    and so `x`'s type might or might not change.

    Returns:
        A list of numpy arrays with the same length as `array_to_match`.
    """
    if len(array_to_match) < len(array_to_pad):
        raise ValueError(
            f"Length of 'array_to_match' ({len(array_to_match)}) must "
            f"be >= 'array_to_pad' ({len(array_to_pad)})"
        )

    def pad_one_array(
        to_pad,
        template,
    ):
        delta_length = template.shape[0] - to_pad.shape[0]
        if delta_length == 0:
            return to_pad
        elif delta_length < 0:
            return to_pad[0 : template.shape[0]]  # noqa: E203

        # Fill in extra values
        npad = [(0, delta_length)] + [(0, 0) for _ in range(to_pad.ndim - 1)]
        return np.pad(to_pad, pad_width=tuple(npad), mode="symmetric")

    return [pad_one_array(adv, og) for adv, og in zip(array_to_pad, array_to_match)]


def _do_split(arr: List[np.ndarray], n: int) -> List[List[np.ndarray]]:
    """
    Splits a list of numpy arrays into n sublists, where each sublist contains
    the corresponding elements of the input arrays split into n parts.

    Args:
        arr (List[np.ndarray]): The list of numpy arrays to split.
        n (int): The number of sublists to split the arrays into.

    Returns:
        List[List[np.ndarray]]: A list of n sublists, where each sublist contains
        the corresponding elements of the input arrays split into n parts.
    """
    arr_s = [np.array_split(to_split, n) for to_split in arr]
    return [[arr_s[i][k] for i in range(len(arr))] for k in range(n)]


def setup_adversary_arrays(
    mcWeights_val_adversary: pd.DataFrame,
    weights_to_train: List[np.ndarray],
    weights_to_validate: List[np.ndarray],
    weights_train_adversary_s: pd.Series,
    weights_val_adversary_orig: pd.Series,
    x_to_adversary_orig: List[np.ndarray],
    x_to_train: List[np.ndarray],
    x_to_validate: List[np.ndarray],
    x_to_validate_adv_orig: List[np.ndarray],
    y_to_train: List[np.ndarray],
    y_to_train_adversary_orig: Union[List[np.ndarray], List[pd.DataFrame]],
    y_to_validate: List[np.ndarray],
    y_to_validate_adv: List[np.ndarray],
    training_params: TrainingConfig,
):
    """Sets up adversary inputs. Must be as long as main inputs, so do some
    repetition if they are different sizes

    :param mcWeights_val_adversary: CR jet mc Weights
    :param weights_to_train: Main Jet weights, training set
    :param weights_to_validate: Main jet weights, validation set
    :param weights_train_adversary: CR jet weights, training set
    :param weights_val_adversary: CR jet weights, validation set
    :param x_to_adversary: input variables, CR jets, training set
    :param x_to_train: input variables, main jets, training set
    :param x_to_validate: input variables, main jets, validation set
    :param x_to_validate_adv: input variables, CR jets, validation set
    :param y_to_train: labels, main jets, training set
    :param y_to_train_adversary: labels, CR jets, training set
    :param y_to_validate: labels, main jets, validation set
    :param y_to_validate_adv: labels, CR jets, validation sets
    :param training_params: class of training parameters
    :return: set up arrays for training
    """
    small_x_to_adversary = x_to_adversary_orig.copy()
    small_y_to_train_adversary = y_to_train_adversary_orig.copy()
    small_weights_train_adversary = weights_train_adversary_s.values.copy()
    small_x_val_adversary = x_to_validate_adv_orig.copy()
    small_y_val_adversary = y_to_validate_adv.copy()
    small_weights_val_adversary = weights_val_adversary_orig.values.copy()
    small_mcWeights_val_adversary = mcWeights_val_adversary.values.copy()

    # Shorten or repeat to get the various arrays the same size. The model
    # design demands that they be the same size. Since the control MC is
    # shorter (often) we will pay with `np.pad``.

    x_to_adversary = _pad_arrays(x_to_adversary_orig, x_to_train)
    y_to_train_adversary = _pad_arrays(y_to_train_adversary_orig, y_to_train)
    x_to_validate_adv = _pad_arrays(x_to_validate_adv_orig, x_to_validate)
    y_to_validate_adv = _pad_arrays(y_to_validate_adv, y_to_validate)

    weights_train_adversary = _pad_arrays(
        [weights_train_adversary_s], weights_to_train
    )[0]

    weights_val_adversary_values = weights_val_adversary_orig.values
    weights_val_adversary = _pad_arrays(
        [weights_val_adversary_values], weights_to_validate
    )[0]

    # Basically how many mini-batches
    # Important as too few, too much data, GPU cannot handle it
    assert training_params.num_splits is not None
    num_splits = training_params.num_splits

    x_to_train_split = _do_split(x_to_train, num_splits)
    x_to_adversary_split = _do_split(x_to_adversary, num_splits)

    y_to_train_0 = np.array_split(y_to_train[0], num_splits)

    y_to_train_adversary_squeeze = np.array_split(
        np.squeeze(np.array(y_to_train_adversary)), num_splits
    )

    weights_to_train_0 = np.array_split(weights_to_train[0], num_splits)

    weights_train_adversary = np.array_split(weights_train_adversary, num_splits)
    small_x_to_adversary_split = _do_split(small_x_to_adversary, num_splits)

    small_y_to_train_adversary_0 = np.array_split(
        small_y_to_train_adversary[0], num_splits
    )
    assert isinstance(small_weights_train_adversary, np.ndarray)
    small_weights_train_adversary_s = np.array_split(
        small_weights_train_adversary, num_splits
    )

    num_splits_adv = num_splits

    x_to_validate_split = _do_split(x_to_validate, num_splits_adv)
    x_to_validate_adv_split = _do_split(x_to_validate_adv, num_splits_adv)

    y_to_validate_0 = np.array_split(y_to_validate[0], num_splits_adv)
    y_to_validate_adv_squeeze = np.array_split(
        np.squeeze(np.array(y_to_validate_adv)), num_splits_adv
    )
    weights_to_validate_0 = np.array_split(weights_to_validate[0], num_splits_adv)

    assert isinstance(weights_val_adversary, np.ndarray)
    weights_val_adversary_split = np.array_split(weights_val_adversary, num_splits_adv)

    return (
        num_splits,
        small_mcWeights_val_adversary,
        small_weights_train_adversary_s,
        small_weights_val_adversary,
        small_x_to_adversary_split,
        small_x_val_adversary,
        small_y_to_train_adversary_0,
        small_y_val_adversary,
        weights_to_train_0,
        weights_to_validate_0,
        weights_train_adversary,
        weights_val_adversary_split,
        x_to_adversary_split,
        x_to_train_split,
        x_to_validate_adv_split,
        x_to_validate_split,
        y_to_train_0,
        y_to_train_adversary_squeeze,
        y_to_validate_0,
        y_to_validate_adv_squeeze,
    )
