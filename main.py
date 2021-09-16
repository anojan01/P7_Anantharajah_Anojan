# streamlit run C:/Users/anoja/PycharmProjects/Projet_7/main.py

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from lightgbm import LGBMClassifier

X = pd.read_csv("X_train.csv")

st.write("# Application permettant de prédire l'accord à un prêt bancaire")

# Collecter le profil du profil

st.sidebar.header("Les caractéristiques du client")

def client_carac():

    Annee_naissance = st.sidebar.selectbox('Année de Naissance',range(1900,2021))
    Annee_travail = st.sidebar.selectbox('Année du premier emploi',range(1900,2021))
    EXT_SOURCE_1 = st.sidebar.number_input('EXT_SOURCE_1',-100000.,100000.,0.)
    EXT_SOURCE_2 = st.sidebar.number_input('EXT_SOURCE_2',-100000.,100000.,0.)
    EXT_SOURCE_3 = st.sidebar.number_input('EXT_SOURCE_3',-100000.,100000.,0.)
    AMT_BALANCE = st.sidebar.number_input('AMT_BALANCE',-100000.,100000.,0.)
    AMT_ANNUITY = st.sidebar.number_input('AMT_ANNUITY',-100000.,100000.,0.)
    AMT_DRAWINGS_CURRENT = st.sidebar.number_input('AMT_DRAWINGS_CURRENT',-100000.,100000.,0.)
    AMT_CREDIT_MAX_OVERDUE = st.sidebar.number_input('AMT_CREDIT_MAX_OVERDUE',-100000.,100000.,0.)
    CREDIT_DAY_OVERDUE = st.sidebar.number_input('CREDIT_DAY_OVERDUE',-100000.,100000.,0.)
    DAYS_CREDIT_ENDDATE = st.sidebar.number_input('DAYS_CREDIT_ENDDATE',-100000.,100000.,0.)
    SK_DPD = st.sidebar.number_input('SK_DPD',-100000.,100000.,0.)
    NONLIVINGAPARTMENTS_MODE = st.sidebar.number_input('NONLIVINGAPARTMENTS_MODE',-100000.,100000.,0.)
    YEARS_BUILD_MODE = st.sidebar.number_input('YEARS_BUILD_MODE',-100000.,100000.,0.)
    ENTRANCES_MODE = st.sidebar.number_input('ENTRANCES_MODE',-100000.,100000.,0.)
    BASEMENTAREA_MODE = st.sidebar.number_input('BASEMENTAREA_MODE',0.,100.,0.)
    OWN_CAR_AGE = st.sidebar.number_input('OWN_CAR_AGE',-100000.,100000.,0.)
    PERIOD = st.sidebar.number_input('PERIOD',-100000.,100000.,0.)
    REG_CITY_NOT_WORK_CITY = st.sidebar.number_input('REG_CITY_NOT_WORK_CITY',-100000.,100000.,0.)
    CNT_PAYMENT = st.sidebar.number_input('CNT_PAYMENT',0.,100.,0.)
    NFLAG_INSURED_ON_APPROVAL = st.sidebar.number_input('NFLAG_INSURED_ON_APPROVAL',-100000.,100000.,0.)


    data = {
        "Nb J avant premier emploi": (Annee_travail - Annee_naissance)*365,
        "EXT_SOURCE_1" : EXT_SOURCE_1,
        "EXT_SOURCE_2" : EXT_SOURCE_2,
        "EXT_SOURCE_3" : EXT_SOURCE_3,
        "AMT_BALANCE" : AMT_BALANCE,
        "AMT_ANNUITY" : AMT_ANNUITY,
        "AMT_DRAWINGS_CURRENT" : AMT_DRAWINGS_CURRENT,
        "AMT_CREDIT_MAX_OVERDUE" : AMT_CREDIT_MAX_OVERDUE,
        "CREDIT_DAY_OVERDUE" : CREDIT_DAY_OVERDUE,
        "DAYS_CREDIT_ENDDATE" : DAYS_CREDIT_ENDDATE,
        "SK_DPD" : SK_DPD,
        "NONLIVINGAPARTMENTS_MODE" : NONLIVINGAPARTMENTS_MODE,
        "YEARS_BUILD_MODE" : YEARS_BUILD_MODE,
        "ENTRANCES_MODE" : ENTRANCES_MODE,
        "BASEMENTAREA_MODE" : BASEMENTAREA_MODE,
        "OWN_CAR_AGE" : OWN_CAR_AGE,
        "PERIOD" : PERIOD,
        "REG_CITY_NOT_WORK_CITY" : REG_CITY_NOT_WORK_CITY,
        "CNT_PAYMENT" : CNT_PAYMENT,
        "NFLAG_INSURED_ON_APPROVAL" : NFLAG_INSURED_ON_APPROVAL

    }

    profil = pd.DataFrame(data, index=[0])
    return profil

input_df = client_carac()

#Transformation données d'entrées

st.subheader("Synthèse Client")
st.write(input_df)

# Import Modele

model = pickle.load(open("modele_final.pkl","rb"))

prevision = model.predict(input_df)

st.subheader("Resultat de la prévision")
st.write(prevision)

if prevision == 1 :
    st.subheader("Prêt accordé")
else :
    st.subheader("Prêt non accordé")