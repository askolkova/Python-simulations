#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 14:08:46 2018

@author: AS
""" 
# Description: here I included both OLS with JK and RIDGE (yet without JK), 
#               and compare the perfomance of averaged estimators out of 
#               sample;
# modifies Ridge_OLS_simulations_NO_testsample_v2.py by adding MMA
#
# NB: now there is a problem with infeasible OLS


# https://matplotlib.org/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py

import numpy as np
import quadprog
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from scipy.special import zeta

np.random.seed(123)

### DATA GENERATION
alpha = 0.5
n1 = 50 # number of obs 
repl = 1000 # number of replications
R_sq_cases = 10 # how many diff values of R2 to try
J = 70 # "length" of the true model
OLS_AMSE = []
OLS_AMSE_norm = []
infeasible_AMSE = []
#RIDGE_AMSE = []
#MALLOW_AMSE = []
OLS_weights_aver = []
#RIDGE_weights_aver = []
#MALLOW_weights_aver = []

theta_j_R = np.zeros((J, R_sq_cases))
c1_sq_of_R = np.zeros((R_sq_cases,1))

split_n = 50
n1_in = n1

M = int(np.round( 3. * n1_in**(1./3) )) # M=11, number of models for n1
K = np.arange(1,M+1) # vector with number of parameters in models, from 1 to M
# will be used for construction of MMA

n_lambdas = 100 # tuning/shrinkage parameter
lambdas = np.logspace(-2, 2, n_lambdas)

R_sq_grid = np.linspace(0.1, 0.8, R_sq_cases)
for R_sq_ind in range(R_sq_cases): #start iteration across different R^2
    R_sq = R_sq_grid[R_sq_ind]
    #c1 = arnp.sqrt(R_sq / (1 - R_sq))
    c1 = np.sqrt(R_sq/ (2. * alpha * zeta(1 + 2.*alpha) - R_sq * zeta(1 + 2.*alpha)))
    c1_sq_of_R[R_sq_ind,0] = c1**2/(1+c1**2)
    theta_j = [1]
    
    for j in range(1,J):
        theta_j.append(c1 * np.sqrt(2*alpha) * j**(-alpha - 0.5))
             
    theta_j_R[:,R_sq_ind] = np.array(theta_j).reshape(J,)
    
    OLS_weights = np.zeros((M, repl)) # vectors of opt weights, for every replication
    #RIDGE_weights = np.zeros((M, repl))
#    MALLOW_weights = np.zeros((M, repl))
    
    OLS_y_hat_weighted_in = np.zeros((n1_in, repl)) # vector of predicted values, for every simulation
    #OLS_y_hat_weighted_out = np.zeros((n1_out, repl))
    #RIDGE_y_hat_weighted_in = np.zeros((n1_in, repl))
    #RIDGE_y_hat_weighted_out = np.zeros((n1_out, repl))
#    MALLOW_y_hat_weighted_in  = np.zeros((n1_in, repl))
    
    
    OLS_MSE_in = np.zeros((repl, 1)) # for mean sq. error
    OLS_MSE_in_norm = np.zeros((repl, 1))
    #OLS_MSE_out = np.zeros((repl, 1))
    #RIDGE_MSE_in = np.zeros((repl, 1))
    #RIDGE_MSE_out = np.zeros((repl, 1))
#    MALLOW_MSE_in = np.zeros((repl, 1))
    
    MSE_infeasible_OLS = np.zeros((repl, 1))
    
    for r in range(1, repl+1): # start iterate across replications
        if r % 50 == 0:
            print('R_sq_ind = '+str(R_sq_ind)+',repl_numb = ' + str(r))
        epsilon = np.random.randn(n1,1) # error term, homosked
        X = np.ones((n1,1)) # for the constant term
        # drop X = np.concatenate((X, np.random.randn(n1, J-1)), axis=1) #### !!!!
        # x-s are iid N(0,1)
        dimen = max(J, n1)
        X = np.concatenate((X, np.random.randn(n1, dimen)), axis=1)
        Y1 = np.dot(X[:,0:J], np.array(theta_j)) + epsilon.reshape(n1,)
        # drop Y1 = np.dot(X, np.array(theta_j) ) + epsilon.reshape(n1,)
        Y1 = Y1.reshape(n1,1)
        ### READY TO DO RIDGE and OLS
        OLS_resids_m_in = np.zeros((n1_in, M)) # for OLS residuals
        JK_resids_m_in = np.zeros((n1_in, M)) # for JK OLS resids
        #RIDGE_resids_m_in = np.zeros((n1_in, M)) 
        #JK_resids_m = np.zeros((n1, M)) # for JK residual which are easy to to obtain from OLS res-s
        OLS_y_m_hat_in = np.zeros((n1_in, M))
        #OLS_y_m_hat_out = np.zeros((n1_out, M))
        #RIDGE_y_m_hat_in = np.zeros((n1_in, M))
        #RIDGE_y_m_hat_out = np.zeros((n1_out, M))
        
        OLS_beta_m = np.zeros((M,M))
        #RIDGE_beta_m = np.zeros((M,M))
        
        # in-sample portion
        X_in = X[0:split_n,:]
        Y1_in = Y1[0:split_n]
        # out-of-sample-portion
        #X_out = X[split_n:n1+1,:]
        #Y1_out= Y1[split_n:n1+1]
        
        for m in range(1,M+1): 
            # RIDGE part
#            ridgeCV = linear_model.RidgeCV(lambdas, fit_intercept=False, normalize=False, scoring=None, cv=None, gcv_mode=None, store_cv_values=False)
#            ridgeCV.fit(X_in[:, 0:m], Y1_in)
#            RIDGE_beta_m[0:m, m-1:m] = ridgeCV.coef_.reshape(m,1)
#            # find the predicted values and residuals
#            RIDGE_y_m_hat_in[:, m-1:m] = ridgeCV.predict(X_in[:, 0:m]) # in-sample prediction
#            #RIDGE_y_m_hat_out[:, m-1:m] = ridgeCV.predict(X_out[:, 0:m]) # out-of-sample prediction
#            RIDGE_resids_m_in[:, m-1:m] = Y1_in - RIDGE_y_m_hat_in[:, m-1:m].reshape(n1_in, 1)
            
            #OLS part
            accuracy_mat = np.linalg.inv(np.dot(np.transpose(X_in[:, 0:m]), X_in[:, 0:m]))
            proj_mat = np.dot(np.dot(X_in[:, 0:m], accuracy_mat), np.transpose(X_in[:, 0:m]))
            OLS_beta_m[0:m, m-1:m] = np.dot(np.dot(accuracy_mat, np.transpose(X_in[:, 0:m])) , Y1_in)
            OLS_y_m_hat_in[:, m-1:m] = np.dot(proj_mat, Y1_in)
            #OLS_y_m_hat_out[:,m-1:m] = np.dot(X_out[:, 0:m], OLS_beta_m[0:m, m-1:m])
            hs_m = np.diag(proj_mat, k=0) # getting n diagonal elements from P_m
            D_m = np.zeros((n1_in, n1_in))
            for ind_n in range(n1_in):
                D_m[ind_n, ind_n] = 1 / (1 - hs_m[ind_n])
            OLS_resids_m_in[:, m-1:m] = Y1_in - np.dot(proj_mat, Y1_in).reshape(n1_in,1)
            JK_resids_m_in[:, m-1:m] = np.dot(D_m, OLS_resids_m_in[:, m-1:m])
            
         # obtain best / infeasible predictions and MSE
#        accuracy_mat = np.linalg.inv(np.dot(np.transpose(X_in[:, 0:J]), X_in[:, 0:J]))
#        proj_mat = np.dot(np.dot(X_in[:, 0:J], accuracy_mat), np.transpose(X_in[:, 0:J]))
#        OLS_y_m_hat_in_infeasible = np.dot(proj_mat, Y1_in)   
        
        # different wway to do infeasible OLS
        OLS_y_m_hat_in_infeasible = np.dot(X_in[:,0:J], np.array(theta_j))
            
        OLS_S_n = np.dot(np.transpose(JK_resids_m_in), JK_resids_m_in)/(n1_in) # criterion to min
        #OLS_S_n =  np.dot(np.transpose(OLS_resids_m_in), OLS_resids_m_in)/(n1_in) # criterion to min
#        RIDGE_S_n = np.dot(np.transpose(RIDGE_resids_m_in), RIDGE_resids_m_in)/(n1_in) 
#        MALLOW_S_n = np.dot(np.transpose(OLS_resids_m_in), OLS_resids_m_in)/(n1_in)
        
        #SOLVE QP problem
        # ---- OLS --- 
        G = OLS_S_n * 2
        a = np.zeros((M,1)).reshape(M,)
        C1 = np.ones((1, M))
        C2 = np.eye(M, dtype=int)
        Constraints = np.concatenate((C1, C2), axis=0).T # matrix defining the constraints
        b = np.eye(1, dtype=int)
        b = np.concatenate((b, np.zeros((M,1))), axis=0).reshape(M+1,)# vector defining the constraints  
        
        x = quadprog.solve_qp(G, a, Constraints, b, meq=1) # JK
        OLS_weights[:, r-1:r] = x[0].reshape(M, 1) 
        # if small negative, then 0
        OLS_weights[:, r-1:r] = OLS_weights[:, r-1:r] * (OLS_weights[:, r-1:r] > 0)
        OLS_y_hat_weighted_in[:, r-1:r] = np.dot(OLS_y_m_hat_in, OLS_weights[:, r-1:r]) # get the weighted est from non-weighted ones
        OLS_MSE_in[r-1:r,0] = np.square(Y1_in - OLS_y_hat_weighted_in[:, r-1:r]).mean()
    
        MSE_infeasible_OLS[r-1:r,0] = np.square(Y1_in.reshape(50,) - OLS_y_m_hat_in_infeasible).mean()
        OLS_MSE_in_norm[r-1:r,0] = OLS_MSE_in[r-1:r,0]/MSE_infeasible_OLS[r-1:r,0]
        
        #OLS_y_hat_weighted_out[:, r-1:r] = np.dot(OLS_y_m_hat_out, OLS_weights[:, r-1:r]) 
        #OLS_MSE_out[r-1:r,0] = np.square(Y1_out - OLS_y_hat_weighted_out[:, r-1:r]).mean()
        #MSE_infeasible_OLS_out = np.square(Y1_out - OLS_y_m_hat_out[:, J-1:J]).mean()
        #OLS_MSE_out[r-1:r,0] = OLS_MSE_out[r-1:r,0]/MSE_infeasible_OLS_out
        
        # ---- RIDGE ---- 
#        G = RIDGE_S_n * 2
#        x = quadprog.solve_qp(G, a, Constraints, b, meq=1)
#        RIDGE_weights[:, r-1:r] = x[0].reshape(M, 1) # optimal  weights
#        # if small negative, then 0
#        RIDGE_weights[:, r-1:r] = RIDGE_weights[:, r-1:r] * (RIDGE_weights[:, r-1:r] > 0)
#        RIDGE_y_hat_weighted_in[:, r-1:r] = np.dot(RIDGE_y_m_hat_in, RIDGE_weights[:, r-1:r])
#        RIDGE_MSE_in[r-1:r,0] = np.square(Y1_in - RIDGE_y_hat_weighted_in[:, r-1:r]).mean()
#        RIDGE_MSE_in[r-1:r,0] = RIDGE_MSE_in[r-1:r,0]/MSE_infeasible_OLS
#        
        #RIDGE_y_hat_weighted_out[:, r-1:r] = np.dot(RIDGE_y_m_hat_out, RIDGE_weights[:, r-1:r])
        #RIDGE_MSE_out[r-1:r,0] = np.square(Y1_out - RIDGE_y_hat_weighted_out[:, r-1:r]).mean()
        #RIDGE_MSE_out[r-1:r,0] = RIDGE_MSE_out[r-1:r,0]/MSE_infeasible_OLS_out
        
        # ---- MALLOW ----
#        sigma_sq_est = sum(OLS_resids_m_in[:, M-1] ** 2)/ (n1_in - M)
#        G = MALLOW_S_n * 2
#        a = K.reshape(M,)*(-2)*sigma_sq_est
#        x = quadprog.solve_qp(G, a, Constraints, b, meq=1)
#        MALLOW_weights[:, r-1:r] = x[0].reshape(M, 1) # optimal  weights
#        # if small negative, then 0
#        MALLOW_weights[:, r-1:r] = MALLOW_weights[:, r-1:r] * (MALLOW_weights[:, r-1:r] > 0)
#        MALLOW_y_hat_weighted_in[:, r-1:r]  = np.dot(OLS_y_m_hat_in, MALLOW_weights[:, r-1:r])
#        MALLOW_MSE_in[r-1:r,0] = np.square(Y1_in - MALLOW_y_hat_weighted_in[:, r-1:r]).mean()
#        MALLOW_MSE_in[r-1:r,0] = MALLOW_MSE_in[r-1:r,0] / MSE_infeasible_OLS
        
    infeasible_AMSE.append(MSE_infeasible_OLS.mean())    
    OLS_AMSE.append(OLS_MSE_in.mean())# here I have AMSE as a function of R_sq
    OLS_AMSE_norm.append(OLS_MSE_in_norm.mean())
#    RIDGE_AMSE.append(RIDGE_MSE_in.mean())
#    MALLOW_AMSE.append(MALLOW_MSE_in.mean())
    
    OLS_weights_aver.append(np.mean(OLS_weights, axis = 1))
#    RIDGE_weights_aver.append(np.mean(RIDGE_weights, axis = 1))
#    MALLOW_weights_aver.append(np.mean(MALLOW_weights, axis = 1))

plt.figure(figsize=(10, 4))
plt.subplot(111)    
ax = plt.gca()
ax.plot(np.linspace(0.1, 0.8, R_sq_cases),np.array(OLS_AMSE), "bo--", linewidth=3, markersize=10, label="OLS")
ax.plot(np.linspace(0.1, 0.8, R_sq_cases),np.array(OLS_AMSE_norm), "o-", linewidth=3, markersize=10, label="OLS norm", color = "goldenrod")
ax.plot(np.linspace(0.1, 0.8, R_sq_cases),np.array(infeasible_AMSE), "o--", linewidth=3, markersize=10, label="infeasible OLS ", color = "olivedrab")

#ax.plot(np.linspace(0.1, 0.8, R_sq_cases),np.array(RIDGE_AMSE), "r*-", linewidth=3, markersize=10, label="RIDGE")
#ax.plot(np.linspace(0.1, 0.8, R_sq_cases),np.array(MALLOW_AMSE), "g^-.", linewidth=3, markersize=10, label="MALLOW")
#ax.legend(['OLS','RIDGE'])
ax.legend(fontsize=14)
#ax.plot(np.arange(1,10)/10., c1_sq_of_R)
#ax.set_xscale('log')
#ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('R-squared', fontsize=14)
plt.ylabel('MSE', fontsize=14)
plt.title('MSE as a function of R-squared, n=50', fontsize=16)
plt.ylim(0.8, 1.8) 


"""fig = plt.figure(figsize=(10, 21))
ax = fig.add_subplot(311)
indices = np.arange(1,M+1)
width = np.min(np.diff(indices))/3.
ax.bar(indices - 0.6*width, OLS_weights_aver[0], width=0.25, align='edge',label='OLS', color='y')
#ax.bar(indices, RIDGE_weights_aver[0], width=0.25, align='edge', label='Ridge', color='g')
ax.bar(indices + 0.6*width, MALLOW_weights_aver[0], width=0.25, align='edge', label='Mallow', color='black')
#ax.axes.set_xticklabels(np.arange(1, 12, 1),fontsize=16)
plt.xticks(np.arange(1, 12, 1), fontsize=14)
plt.legend(fontsize=14)
#plt.legend((p1[0], p2[0], p3[0]), ('OLS', 'RIDGE', 'MALLOW'))
t1 = plt.xlabel('Length of the model', fontsize=14, color='black')
t2 = plt.ylabel('Weights', fontsize=14, color='black')
t3 = plt.title('Scores by group and gender', fontsize=16)

ax = fig.add_subplot(312)
ax.bar(indices - 0.6*width, OLS_weights_aver[4], width=0.25, align='edge',label='OLS', color='y')
#ax.bar(indices, RIDGE_weights_aver[4], width=0.25, align='edge', label='Ridge', color='g')
ax.bar(indices + 0.6*width, MALLOW_weights_aver[4], width=0.25, align='edge', label='Mallow', color='black')
#ax.axes.set_xticklabels(np.arange(1, 12, 1),fontsize=16)
plt.xticks(np.arange(1, 12, 1), fontsize=14)
plt.legend(fontsize=14)
#plt.legend((p1[0], p2[0], p3[0]), ('OLS', 'RIDGE', 'MALLOW'))
t1 = plt.xlabel('Length of the model', fontsize=14, color='black')
t2 = plt.ylabel('Weights', fontsize=14, color='black')
t3 = plt.title('Scores by group and gender', fontsize=16)

ax = fig.add_subplot(313)
ax.bar(indices - 0.6*width, OLS_weights_aver[9], width=0.25, align='edge',label='OLS', color='y')
#ax.bar(indices, RIDGE_weights_aver[9], width=0.25, align='edge', label='Ridge', color='g')
ax.bar(indices + 0.6*width, MALLOW_weights_aver[9], width=0.25, align='edge', label='Mallow', color='black')
#ax.axes.set_xticklabels(np.arange(1, 12, 1),fontsize=16)
plt.xticks(np.arange(1, 12, 1), fontsize=14)
plt.legend(fontsize=14)
#plt.legend((p1[0], p2[0], p3[0]), ('OLS', 'RIDGE', 'MALLOW'))
t1 = plt.xlabel('Length of the model', fontsize=14, color='black')
t2 = plt.ylabel('Weights', fontsize=14, color='black')
t3 = plt.title('Scores by group and gender', fontsize=16)
"""