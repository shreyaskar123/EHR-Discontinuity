#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
medicare = pd.read_csv("/netapp2/home/se197/data/CMS/Data/medicare.csv")


# In[1]:


get_ipython().system('pip install xgboost')


# In[2]:


train_set = medicare[medicare.Hospital != 'BWH'] # MGH; n = 204014
validation_set = medicare[medicare.Hospital == 'BWH'] # BWH and Neither; n = 115726
import numpy as np

fifty_perc_EHR_cont = np.percentile(medicare['Cal_MPEC_R0'],50)
train_set_high = train_set[train_set.Cal_MPEC_R0 >= fifty_perc_EHR_cont]
train_set_low= train_set[train_set.Cal_MPEC_R0 < fifty_perc_EHR_cont]

validation_set_high = validation_set[validation_set.Cal_MPEC_R0 >= fifty_perc_EHR_cont]
validation_set_low = validation_set[validation_set.Cal_MPEC_R0 < fifty_perc_EHR_cont]


# In[3]:


predictor_variable = [
        'Co_CAD_R0', 'Co_Embolism_R0', 'Co_DVT_R0', 'Co_PE_R0', 'Co_AFib_R0',
        'Co_Hypertension_R0', 'Co_Hyperlipidemia_R0', 'Co_Atherosclerosis_R0',
        'Co_HF_R0', 'Co_HemoStroke_R0', 'Co_IscheStroke_R0', 'Co_OthStroke_R0',
        'Co_TIA_R0', 'Co_COPD_R0', 'Co_Asthma_R0', 'Co_Pneumonia_R0', 'Co_Alcoholabuse_R0',
        'Co_Drugabuse_R0', 'Co_Epilepsy_R0', 'Co_Cancer_R0', 'Co_MorbidObesity_R0',
        'Co_Dementia_R0', 'Co_Depression_R0', 'Co_Bipolar_R0', 'Co_Psychosis_R0',
        'Co_Personalitydisorder_R0', 'Co_Adjustmentdisorder_R0', 'Co_Anxiety_R0',
        'Co_Generalizedanxiety_R0', 'Co_OldMI_R0', 'Co_AcuteMI_R0', 'Co_PUD_R0',
        'Co_UpperGIbleed_R0', 'Co_LowerGIbleed_R0', 'Co_Urogenitalbleed_R0',
        'Co_Othbleed_R0', 'Co_PVD_R0', 'Co_LiverDisease_R0', 'Co_MRI_R0',
        'Co_ESRD_R0', 'Co_Obesity_R0', 'Co_Sepsis_R0', 'Co_Osteoarthritis_R0',
        'Co_RA_R0', 'Co_NeuroPain_R0', 'Co_NeckPain_R0', 'Co_OthArthritis_R0',
        'Co_Osteoporosis_R0', 'Co_Fibromyalgia_R0', 'Co_Migraine_R0', 'Co_Headache_R0',
        'Co_OthPain_R0', 'Co_GeneralizedPain_R0', 'Co_PainDisorder_R0',
        'Co_Falls_R0', 'Co_CoagulationDisorder_R0', 'Co_WhiteBloodCell_R0', 'Co_Parkinson_R0',
        'Co_Anemia_R0', 'Co_UrinaryIncontinence_R0', 'Co_DecubitusUlcer_R0',
        'Co_Oxygen_R0', 'Co_Mammography_R0', 'Co_PapTest_R0', 'Co_PSATest_R0',
        'Co_Colonoscopy_R0', 'Co_FecalOccultTest_R0', 'Co_FluShot_R0', 'Co_PneumococcalVaccine_R0', 'Co_RenalDysfunction_R0', 'Co_Valvular_R0', 'Co_Hosp_Prior30Days_R0',
        'Co_RX_Antibiotic_R0', 'Co_RX_Corticosteroid_R0', 'Co_RX_Aspirin_R0', 'Co_RX_Dipyridamole_R0',
        'Co_RX_Clopidogrel_R0', 'Co_RX_Prasugrel_R0', 'Co_RX_Cilostazol_R0', 'Co_RX_Ticlopidine_R0',
        'Co_RX_Ticagrelor_R0', 'Co_RX_OthAntiplatelet_R0', 'Co_RX_NSAIDs_R0',
        'Co_RX_Opioid_R0', 'Co_RX_Antidepressant_R0', 'Co_RX_AAntipsychotic_R0', 'Co_RX_TAntipsychotic_R0',
        'Co_RX_Anticonvulsant_R0', 'Co_RX_PPI_R0', 'Co_RX_H2Receptor_R0', 'Co_RX_OthGastro_R0',
        'Co_RX_ACE_R0', 'Co_RX_ARB_R0', 'Co_RX_BBlocker_R0', 'Co_RX_CCB_R0', 'Co_RX_Thiazide_R0',
        'Co_RX_Loop_R0', 'Co_RX_Potassium_R0', 'Co_RX_Nitrates_R0', 'Co_RX_Aliskiren_R0',
        'Co_RX_OthAntihypertensive_R0', 'Co_RX_Antiarrhythmic_R0', 'Co_RX_OthAnticoagulant_R0',
        'Co_RX_Insulin_R0', 'Co_RX_Noninsulin_R0', 'Co_RX_Digoxin_R0', 'Co_RX_Statin_R0',
        'Co_RX_Lipid_R0', 'Co_RX_Lithium_R0', 'Co_RX_Benzo_R0', 'Co_RX_ZDrugs_R0',
        'Co_RX_OthAnxiolytic_R0', 'Co_RX_Barbiturate_R0', 'Co_RX_Dementia_R0', 'Co_RX_Hormone_R0',
        'Co_RX_Osteoporosis_R0', 'Co_N_Drugs_R0', 'Co_N_Hosp_R0', 'Co_Total_HospLOS_R0',
        'Co_N_MDVisit_R0', 'Co_RX_AnyAspirin_R0', 'Co_RX_AspirinMono_R0', 'Co_RX_ClopidogrelMono_R0',
        'Co_RX_AspirinClopidogrel_R0', 'Co_RX_DM_R0', 'Co_RX_Antipsychotic_R0'
]

co_train_gpop = train_set[predictor_variable]
co_train_high = train_set_high[predictor_variable]
co_train_low = train_set_low[predictor_variable]

co_validation_gpop = validation_set[predictor_variable]
co_validation_high = validation_set_high[predictor_variable]
co_validation_low = validation_set_low[predictor_variable]


# In[4]:


# THIS IS WITH NO DEATH
out_train_cardio_gpop = train_set['Out_comp_cardiovascular_nd_RC1']
out_train_cardio_high = train_set_high['Out_comp_cardiovascular_nd_RC1']
out_train_cardio_low = train_set_low['Out_comp_cardiovascular_nd_RC1']

out_validation_cardio_gpop = validation_set['Out_comp_cardiovascular_nd_RC1']
out_validation_cardio_high = validation_set_high['Out_comp_cardiovascular_nd_RC1']
out_validation_cardio_low = validation_set_low['Out_comp_cardiovascular_nd_RC1']

"""
out_train_cardio_gpop = train_set['Out_comp_cardiovascular_R1']
out_train_cardio_high = train_set_high['Out_comp_cardiovascular_R1']
out_train_cardio_low = train_set_low['Out_comp_cardiovascular_R1']

out_validation_cardio_gpop = validation_set['Out_comp_cardiovascular_R1']
out_validation_cardio_high = validation_set_high['Out_comp_cardiovascular_R1']
out_validation_cardio_low = validation_set_low['Out_comp_cardiovascular_R1']
"""


# In[5]:


'''
NOT USING THIS
INSTEAD USING XGBOOST: A FASTER IMPLEMENTATION OF Gradient Boost  
https://github.com/dmlc/xgboost/tree/master/python-package
def GBT(X,y): 
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingRegressor
    from imblearn.over_sampling import SMOTE

    param_grid = [{
        'learning_rate': [0.05,0.1,0.2],
        'n_estimators': [100,150,200]
    }]
    
    boost_clf = GradientBoostingRegressor()
    boosting_grid_search = GridSearchCV(estimator = boost_clf, param_grid = param_grid)
    best_clf = boosting_grid_search.fit(X, y)
    return best_clf
'''


# In[6]:


def xgBoost(X,y):
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV
    model = XGBClassifier()
    param_grid = [{
        'max_depth': [2,3],
        'n_estimators': [60,160],
    }]
    grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    n_jobs = 1,
    cv = 5,
    verbose=True
)
    best_clf = grid_search.fit(X,y)
    return best_clf


# In[7]:


def scores(X,y):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import fbeta_score
    from sklearn.metrics import roc_auc_score 
    from sklearn.metrics import log_loss

    pred = best_clf.predict(X)
    actual = y
    print(accuracy_score(actual,pred), file = open('comp_smote_gbt_ehr.out', 'a'))
    print(f1_score(actual,pred), file = open('comp_smote_gbt_ehr.out', 'a'))
    print(fbeta_score(actual,pred, average = 'macro', beta = 2), file = open('comp_smote_gbt_ehr.out', 'a'))
    print(roc_auc_score(actual, best_clf.predict_proba(X)[:,1]),  file = open('comp_smote_gbt_ehr.out', 'a'))
    print(log_loss(actual,best_clf.predict_proba(X)[:,1]), file = open('comp_smote_gbt_ehr.out', 'a'))


# In[8]:


def cross_val(X,y):
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import log_loss
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import fbeta_score
    import sklearn
    import numpy as np
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    log_loss = [] 
    auc = [] 
    accuracy = []
    f1 = []
    f2 = [] 
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        model = xgBoost(X_train, y_train)
        prob = model.predict_proba(X)[:,1] # prob is a vector of probabilities 
        pred = np.round(model.predict_proba(X)[:,1]) # pred is the rounded predictions 
        log_loss.append(sklearn.metrics.log_loss(y_test, prob))
        auc.append(sklearn.metrics.roc_auc_score(y_test, prob))
        accuracy.append(sklearn.metrics.accuracy_score(y_test, pred))
        f1.append(sklearn.metrics.f1_score(y_test, pred, average = 'macro'))
        f2.append(fbeta_score(y_test,pred, average = 'macro', beta = 2))
    print(np.mean(accuracy),  file = open('comp_smote_gbt_ehr.out', 'a'))
    print(np.mean(f1),  file = open('comp_smote_gbt_ehr.out', 'a'))
    print(np.mean(f2),  file = open('comp_smote_gbt_ehr.out', 'a'))
    print(np.mean(auc),  file = open('comp_smote_gbt_ehr.out', 'a'))
    print(np.mean(log_loss),  file = open('comp_smote_gbt_ehr.out', 'a'))


# # General Population

# In[ ]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 42)
co_train_gpop_sm,out_train_cardio_gpop_sm = sm.fit_resample(co_train_gpop,out_train_cardio_gpop)

best_clf = xgBoost(co_train_gpop_sm, out_train_cardio_gpop_sm)

print("Train gpop", file = open('comp_smote_gbt_ehr.out', 'a'))

cross_val(co_train_gpop_sm, out_train_cardio_gpop_sm)

print()

print("Original Train gpop",  file = open('comp_smote_gbt_ehr.out', 'a'))

scores(co_train_gpop, out_train_cardio_gpop)

print()

print("Test gpop", file = open('comp_smote_gbt_ehr.out', 'a'))

scores(co_validation_gpop, out_validation_cardio_gpop)


# In[ ]:





# # High Continuity 

# In[9]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 42)
co_train_high_sm,out_train_cardio_high_sm = sm.fit_resample(co_train_high,out_train_cardio_high)

best_clf = xgBoost(co_train_high_sm, out_train_cardio_high_sm)

print("Train high", file = open('comp_smote_gbt_ehr.out', 'a'))

cross_val(co_train_high_sm, out_train_cardio_high_sm)

print()

print("Original Train high", file = open('comp_smote_gbt_ehr.out', 'a'))

scores(co_train_high, out_train_cardio_high)

print()

print("Test high", file = open('comp_smote_gbt_ehr.out', 'a'))

scores(co_validation_high, out_validation_cardio_high)


# # Low Continuity
# 

# In[ ]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 42)
co_train_low_sm,out_train_cardio_low_sm = sm.fit_resample(co_train_low,out_train_cardio_low)

best_clf = xgBoost(co_train_low_sm, out_train_cardio_low_sm)

print("Train low", file = open('comp_smote_gbt_ehr.out', 'a'))

cross_val(co_train_low_sm, out_train_cardio_low_sm)

print()

print("Original Train low", file = open('comp_smote_gbt_ehr.out', 'a'))

scores(co_train_low, out_train_cardio_low)

print()

print("Test low", file = open('comp_smote_gbt_ehr.out', 'a'))

scores(co_validation_low, out_validation_cardio_low)


# In[ ]:





# In[ ]:




