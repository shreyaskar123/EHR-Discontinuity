#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
medicare = pd.read_csv("/netapp2/home/se197/RPDR/Josh Lin/3_EHR_V2/CMS/Data/final_medicare.csv")


# In[4]:


train_set = medicare[medicare.Hospital != 'BWH'] # MGH
validation_set = medicare[medicare.Hospital == 'BWH'] # BWH and Neither 
import numpy as np

fifty_perc_EHR_cont = np.percentile(medicare['Cal_MPEC_R0'],50)
train_set_high = train_set[train_set.Cal_MPEC_R0 >= fifty_perc_EHR_cont]
train_set_low= train_set[train_set.Cal_MPEC_R0 < fifty_perc_EHR_cont]

validation_set_high = validation_set[validation_set.Cal_MPEC_R0 >= fifty_perc_EHR_cont]
validation_set_low = validation_set[validation_set.Cal_MPEC_R0 < fifty_perc_EHR_cont]


# In[5]:


predictor_variable = [
'Co_CAD_RC0', 'Co_Embolism_RC0', 'Co_DVT_RC0', 'Co_PE_RC0', 'Co_AFib_RC0',
        'Co_Hypertension_RC0', 'Co_Hyperlipidemia_RC0', 'Co_Atherosclerosis_RC0',
        'Co_HF_RC0', 'Co_HemoStroke_RC0', 'Co_IscheStroke_RC0', 'Co_OthStroke_RC0',
        'Co_TIA_RC0', 'Co_COPD_RC0', 'Co_Asthma_RC0', 'Co_Pneumonia_RC0', 'Co_Alcoholabuse_RC0',
        'Co_Drugabuse_RC0', 'Co_Epilepsy_RC0', 'Co_Cancer_RC0', 'Co_MorbidObesity_RC0',
        'Co_Dementia_RC0', 'Co_Depression_RC0', 'Co_Bipolar_RC0', 'Co_Psychosis_RC0',
        'Co_Personalitydisorder_RC0', 'Co_Adjustmentdisorder_RC0', 'Co_Anxiety_RC0',
        'Co_Generalizedanxiety_RC0', 'Co_OldMI_RC0', 'Co_AcuteMI_RC0', 'Co_PUD_RC0',
        'Co_UpperGIbleed_RC0', 'Co_LowerGIbleed_RC0', 'Co_Urogenitalbleed_RC0',
        'Co_Othbleed_RC0', 'Co_PVD_RC0', 'Co_LiverDisease_RC0', 'Co_MRI_RC0',
        'Co_ESRD_RC0', 'Co_Obesity_RC0', 'Co_Sepsis_RC0', 'Co_Osteoarthritis_RC0',
        'Co_RA_RC0', 'Co_NeuroPain_RC0', 'Co_NeckPain_RC0', 'Co_OthArthritis_RC0',
        'Co_Osteoporosis_RC0', 'Co_Fibromyalgia_RC0', 'Co_Migraine_RC0', 'Co_Headache_RC0',
        'Co_OthPain_RC0', 'Co_GeneralizedPain_RC0', 'Co_PainDisorder_RC0',
        'Co_Falls_RC0', 'Co_CoagulationDisorder_RC0', 'Co_WhiteBloodCell_RC0', 'Co_Parkinson_RC0',
        'Co_Anemia_RC0', 'Co_UrinaryIncontinence_RC0', 'Co_DecubitusUlcer_RC0',
        'Co_Oxygen_RC0', 'Co_Mammography_RC0', 'Co_PapTest_RC0', 'Co_PSATest_RC0',
        'Co_Colonoscopy_RC0', 'Co_FecalOccultTest_RC0', 'Co_FluShot_RC0', 'Co_PneumococcalVaccine_RC0' , 'Co_RenalDysfunction_RC0', 'Co_Valvular_RC0', 'Co_Hosp_Prior30Days_RC0',
        'Co_RX_Antibiotic_RC0', 'Co_RX_Corticosteroid_RC0', 'Co_RX_Aspirin_RC0', 'Co_RX_Dipyridamole_RC0',
        'Co_RX_Clopidogrel_RC0', 'Co_RX_Prasugrel_RC0', 'Co_RX_Cilostazol_RC0', 'Co_RX_Ticlopidine_RC0',
        'Co_RX_Ticagrelor_RC0', 'Co_RX_OthAntiplatelet_RC0', 'Co_RX_NSAIDs_RC0',
        'Co_RX_Opioid_RC0', 'Co_RX_Antidepressant_RC0', 'Co_RX_AAntipsychotic_RC0', 'Co_RX_TAntipsychotic_RC0',
        'Co_RX_Anticonvulsant_RC0', 'Co_RX_PPI_RC0', 'Co_RX_H2Receptor_RC0', 'Co_RX_OthGastro_RC0',
        'Co_RX_ACE_RC0', 'Co_RX_ARB_RC0', 'Co_RX_BBlocker_RC0', 'Co_RX_CCB_RC0', 'Co_RX_Thiazide_RC0',
        'Co_RX_Loop_RC0', 'Co_RX_Potassium_RC0', 'Co_RX_Nitrates_RC0', 'Co_RX_Aliskiren_RC0',
        'Co_RX_OthAntihypertensive_RC0', 'Co_RX_Antiarrhythmic_RC0', 'Co_RX_OthAnticoagulant_RC0',
        'Co_RX_Insulin_RC0', 'Co_RX_Noninsulin_RC0', 'Co_RX_Digoxin_RC0', 'Co_RX_Statin_RC0',
        'Co_RX_Lipid_RC0', 'Co_RX_Lithium_RC0', 'Co_RX_Benzo_RC0', 'Co_RX_ZDrugs_RC0',
        'Co_RX_OthAnxiolytic_RC0', 'Co_RX_Dementia_RC0', 'Co_RX_Hormone_RC0',
        'Co_RX_Osteoporosis_RC0', 'Co_N_Drugs_RC0', 'Co_N_Hosp_RC0', 'Co_Total_HospLOS_RC0',
        'Co_N_MDVisit_RC0', 'Co_RX_AnyAspirin_RC0', 'Co_RX_AspirinMono_RC0', 'Co_RX_ClopidogrelMono_RC0',
        'Co_RX_AspirinClopidogrel_RC0', 'Co_RX_DM_RC0', 'Co_RX_Antipsychotic_RC0'
]
co_train_gpop = train_set[predictor_variable]
co_train_high = train_set_high[predictor_variable]
co_train_low = train_set_low[predictor_variable]

co_validation_gpop = validation_set[predictor_variable]
co_validation_high = validation_set_high[predictor_variable]
co_validation_low = validation_set_low[predictor_variable]

medicare.shape


# In[6]:


out_train_cardio_gpop = train_set['Out_comp_cardiovascular_RC1']
out_train_cardio_high = train_set_high['Out_comp_cardiovascular_RC1']
out_train_cardio_low = train_set_low['Out_comp_cardiovascular_RC1']

out_validation_cardio_gpop = validation_set['Out_comp_cardiovascular_RC1']
out_validation_cardio_high = validation_set_high['Out_comp_cardiovascular_RC1']
out_validation_cardio_low = validation_set_low['Out_comp_cardiovascular_RC1']


# In[7]:


def lr(X_train, y_train):
    from sklearn.linear_model import Lasso
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler 

    model = LogisticRegression()
    param_grid = [
        {'C' : np.logspace(-4, 4, 20)}
        ]
    clf = GridSearchCV(model, param_grid, cv = 5, verbose = True, n_jobs = -1)
    best_clf = clf.fit(X_train, y_train)
    return best_clf


# In[8]:


def scores(X_train,y_train):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import fbeta_score
    from sklearn.metrics import roc_auc_score 
    from sklearn.metrics import log_loss

    pred = best_clf.predict(X_train)
    actual = y_train
    print(accuracy_score(actual,pred),file = open('cardio_nd_famd_lr_ehrc.out', 'a'))
    print(f1_score(actual,pred),file = open('cardio_nd_famd_lr_ehrc.out', 'a'))
    print(fbeta_score(actual,pred, average = 'macro', beta = 2),file = open('cardio_nd_famd_lr_ehrc.out', 'a'))
    print(roc_auc_score(actual, best_clf.predict_proba(X_train)[:,1]),file = open('cardio_nd_famd_lr_ehrc.out', 'a'))
    print(log_loss(actual,best_clf.predict_proba(X_train)[:,1]),file = open('cardio_nd_famd_lr_ehrc.out', 'a'))
    
    


# In[9]:


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
        model = lr(X_train, y_train)
        prob = model.predict_proba(X_test)[:,1] # prob is a vector of probabilities 
        print(prob)
        pred = np.round(prob) # pred is the rounded predictions 
        
        log_loss.append(sklearn.metrics.log_loss(y_test, prob))
        auc.append(sklearn.metrics.roc_auc_score(y_test, prob))
        accuracy.append(sklearn.metrics.accuracy_score(y_test, pred))
        f1.append(sklearn.metrics.f1_score(y_test, pred, average = 'macro'))
        f2.append(fbeta_score(y_test,pred, average = 'macro', beta = 2))
    print(np.mean(accuracy),file = open('cardio_nd_famd_lr_ehrc.out', 'a'))
    print(np.mean(f1),file = open('cardio_nd_famd_lr_ehrc.out', 'a'))
    print(np.mean(f2),file = open('cardio_nd_famd_lr_ehrc.out', 'a'))
    print(np.mean(auc),file = open('cardio_nd_famd_lr_ehrc.out', 'a'))
    print(np.mean(log_loss),file = open('cardio_nd_famd_lr_ehrc.out', 'a'))


# # FAMD Transformation

# In[10]:


from prince import FAMD
famd = FAMD(n_components = 15, n_iter = 3, random_state = 101)

for (colName, colData) in co_train_gpop.iteritems():
    if (colName != 'Co_N_Drugs_R0' and colName!= 'Co_N_Hosp_R0' and colName != 'Co_Total_HospLOS_R0' and colName != 'Co_N_MDVisit_R0'):
        co_train_gpop[colName].replace((1,0) ,('yes','no'), inplace = True)
        co_train_low[colName].replace((1,0) ,('yes','no'), inplace = True)
        co_train_high[colName].replace((1,0) ,('yes','no'), inplace = True)
        co_validation_gpop[colName].replace((1,0), ('yes','no'), inplace = True)
        co_validation_high[colName].replace((1,0), ('yes','no'), inplace = True)
        co_validation_low[colName].replace((1,0), ('yes','no'), inplace = True)




famd.fit(co_train_gpop)
co_train_gpop_FAMD = famd.transform(co_train_gpop)  

famd.fit(co_train_high)
co_train_high_FAMD = famd.transform(co_train_high)  

famd.fit(co_train_low)
co_train_low_FAMD = famd.transform(co_train_low)  

famd.fit(co_validation_gpop)
co_validation_gpop_FAMD = famd.transform(co_validation_gpop)    

famd.fit(co_validation_high)
co_validation_high_FAMD = famd.transform(co_validation_high)    

famd.fit(co_validation_low)
co_validation_low_FAMD = famd.transform(co_validation_low)    


# In[11]:


'''
,
',  , ,
        , , ', 'Co_RX_Hormone_R0',
        'Co_RX_Osteoporosis_R0', 'Co_N_Drugs_R0', 'Co_N_Hosp_R0', 'Co_Total_HospLOS_R0',
        'Co_N_MDVisit_R0', 'Co_RX_AnyAspirin_R0', 'Co_RX_AspirinMono_R0', 'Co_RX_ClopidogrelMono_R0',
        'Co_RX_AspirinClopidogrel_R0', 'Co_RX_DM_R0', 'Co_RX_Antipsychotic_R0'
'''


# # General Population

# In[12]:


print("Gpop",file = open('cardio_nd_famd_lr_ehrc.out', 'a'))
best_clf = lr(co_train_gpop_FAMD, out_train_cardio_gpop)

cross_val(co_train_gpop_FAMD, out_train_cardio_gpop)

print()
print("",file = open('cardio_nd_famd_lr_ehrc.out', 'a'))

scores(co_validation_gpop_FAMD, out_validation_cardio_gpop)


# # Low Continuity

# In[57]:


print("Low",file = open('cardio_nd_famd_lr_ehrc.out', 'a'))
best_clf = lr(co_train_low_FAMD, out_train_cardio_low)

cross_val(co_train_low_FAMD, out_train_cardio_low)

print()
print("",file = open('cardio_nd_famd_lr_ehrc.out', 'a'))

scores(co_validation_low_FAMD, out_validation_cardio_low)


# # High Continuity

# In[58]:


print("High",file = open('cardio_nd_famd_lr_ehrc.out', 'a'))
best_clf = lr(co_train_high_FAMD, out_train_cardio_high)

cross_val(co_train_high_FAMD, out_train_cardio_high)

print("",file = open('cardio_nd_famd_lr_ehrc.out', 'a'))

scores(co_validation_high_FAMD, out_validation_cardio_high)


# In[ ]:




