import networkx as nx
import pandas as pd
import numpy as np
import os
import random

import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler
from sklearn.model_selection import train_test_split

import keras
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score

from stellargraph import globalvar

from sgembeddings import config


def create_train_gen(G):
    # This generates random walk samples from the graph
    unsupervised_samples = UnsupervisedSampler(
        G,
        nodes=list(G.nodes()),
        length=config.WALK_LENGTH,
        number_of_walks=config.NUM_WALKS,
    )

    return GraphSAGELinkGenerator(G, config.BATCH_SIZE, config.NUM_SAMPLES).flow(
        unsupervised_samples
    )


def create_graphsage(train_gen):
    return GraphSAGE(
        layer_sizes=config.LAYER_SIZES,
        generator=train_gen,
        bias=True,
        dropout=config.DROPOUT,
        normalize="l2",
    )

def create_model(graph_sage):
    x_inp, x_out = graph_sage.build(flatten_output=False)

    # classification layer that takes the pair of node embeddings, combines them, puts them
    # through a dense layer
    prediction = link_classification(
        output_dim=1,
        output_act="sigmoid",
        edge_embedding_method="ip",
    )(x_out)

    model = keras.Model(inputs=x_inp, outputs=prediction)

    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

    return x_inp, x_out, model


def train_model(model, train_gen):
    history = model.fit_generator(
        train_gen,
        epochs=config.EPOCHS,
        verbose=1,
        use_multiprocessing=False,
        workers=0,
        shuffle=True,
    )

    return history