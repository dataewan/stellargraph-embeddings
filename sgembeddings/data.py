import networkx as nx
import pandas as pd
import os

import stellargraph as sg

from sgembeddings import config


def read_edgelist():
    edgelist = pd.read_csv(
        os.path.join(config.DATA_DIR, "cora.cites"),
        header=None,
        names=["source", "target"],
        sep="\t",
    )
    edgelist["label"] = "cities"

    return edgelist


def create_graph_from_edgelist(edgelist):
    Gnx = nx.from_pandas_edgelist(edgelist, edge_attr="label")
    nx.set_node_attributes(Gnx, "paper", "label")

    return Gnx


def read_node_features():
    feature_names = ["w_{}".format(ii) for ii in range(1433)]
    column_names = feature_names + ["subject"]
    node_data = pd.read_csv(
        os.path.join(config.DATA_DIR, "cora.content"),
        sep="\t",
        header=None,
        names=column_names,
    )

    node_features = node_data[feature_names]

    return node_data, node_features


def get_graph():
    Gnx = create_graph_from_edgelist(read_edgelist())

    node_data, node_features = read_node_features()

    G = sg.StellarGraph(Gnx, node_features=node_features)
    return node_data, G
