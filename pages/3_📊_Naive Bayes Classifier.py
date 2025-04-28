import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from streamlit_option_menu import option_menu
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Model Penalaran Probabilistik",
    page_icon="📊",
    layout="wide"
)


with st.sidebar:
    selected = option_menu(
        menu_title="MODEL",
        options=["Bayesian Network", "Metode Tambahan", "Fitur Wajib", "Fitur Tambahan"],
    )

if selected == "Bayesian Network":

 