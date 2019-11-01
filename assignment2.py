#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:18:52 2019

@author: aytek
"""

import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def ReadFF(sIn):
    """
    Purpose:
        Read the FF data

    Inputs:
        sIn     string, name of input file

    Return value:
        df      dataframe, data
    """
    df= pd.read_csv(sIn, header=3, names=["Date","Mkt-RF","SMB","HML","RF"])
    df= df.dropna(how='any')

    # Reformat the dates, as date-time, and place them as index
    vDate= pd.to_datetime(df["Date"].values,format='%Y%m%d')
    df.index= vDate

    # Add in a constant
    iN= len(vDate)
    df["C"]= np.ones(iN)

    return df

###########################################################
### JoinStock(df, asStock, sPer)
def JoinStock(df, sStock, sPer):
    """
    Purpose:
        Join the stock into the dataframe, as excess returns

    Inputs:
        df      dataframe, data including RF
        sStock  string, name of stock to read
        sPer    string, extension indicating period (not used)

    Return value:
        df      dataframe, enlarged
    """
    df1= pd.read_csv("data/"+sStock+".csv", index_col="Date", usecols=["Date", "Adj Close"])
    df1.columns= [sStock]

    # Add prices to original dataframe, to get correct dates
    df= df.join(df1, how="left")

    # Extract returns
    vR= 100*np.diff(np.log(df[sStock].values))
    # Add a missing, as one observation was lost differencing
    vR= np.hstack([np.nan, vR])

    # Add excess return to dataframe
    df[sStock + "-RF"]= vR - df["RF"]

    return df

def SaveFF(df, asStock):
    """
    Purpose:
        Prepare data for FF regressions

    Inputs:
        df      dataframe, all data
        asStock list of strings, stocks

    Outputs:
        df to be used in FF regressions
        column list for FF regression
    """


    asOut= ['Mkt-RF', 'SMB', 'HML', 'RF', 'C']
    asOutStocks = []
    for sStock in asStock:
        asOut.append(sStock+"-RF")
        asOutStocks.append(sStock+"-RF")
    print ("Writing ", df.shape[0], " observations on columns ", asOut)
    return df,asOutStocks

def Ksi_Sigma(df,s,factors,nStocks,nFactors):
    """
    Purpose:
        Calculate, ksi, sigma and R squared for each of the stocks

    Inputs:
        df        dataframe, all data
        s         string, stock
        factors   list, list of factors
        nStocks   integer, number of Stocks
        nFactors  integer, number of Factors 

    Outputs:
        ksi       list, ksi vector of the estimation
        sigma     float, variance of the residuals of the estimation
        r_squared float, R squared of the estimation
    """
    cols = factors + list([s])
    df_sub = df[cols].dropna(how='any')
    G = np.array(df_sub[factors])
    R = np.array(df_sub[s])
    
    T = G.shape[0]
    m = G.shape[1]-1
    
    ksi = np.linalg.inv(G.T @ G) @ G.T @ R
    residual = R - G @ ksi
    sigma = residual @ residual.T / (T-m-1)
    
    ##R squared
    R_demeaned = R - np.mean(R)
    r_squared = 1 - (residual @ residual.T)/(R_demeaned.T @ R_demeaned)
    
    return ksi,sigma,r_squared

def main():
    # Magic numbers
    sPer= "0019"
    iY1= 2000
    sIn= "data/F-F_Research_Data_Factors_weekly.CSV"
    asStock= ["aig_0019", "cvx_0019", "msft_0019", "tsla_1019"]
    factors = ['C','Mkt-RF', 'SMB', 'HML']

    # Initialisation
    df= ReadFF(sIn)
    for sStock in asStock:
        df= JoinStock(df, sStock, sPer)
    vI= df.index > str(iY1)
    
    df,cols = SaveFF(df[vI], asStock)
    df.reset_index(drop=True,inplace=True)
    df.dropna(how='any',inplace=True)
    
    asStock  = cols
    #number of stocks and factors
    nStocks = len(asStock)
    nFactors = len(factors)
    
    
    #initialize ksi and sigma vector
    vKsi = np.zeros((nFactors,nStocks))
    dSigma = np.zeros((nStocks,nStocks))
    vRsquared = np.zeros(nStocks)
    # calculate ksi and sigma for each stock and fill the matrix
    for s,j in zip(asStock,range(nStocks)):
        vKsi[:,j],dSigma[j,j],vRsquared[j] = Ksi_Sigma(df,s,factors,nStocks,nFactors)
    
    rSquareds = pd.DataFrame(index = asStock,data=vRsquared,columns=['R^2'])
    print('R-squared of stocks estimations with factor model: \n',rSquareds)
    rSquareds.plot(kind='bar')
    plt.show()
    
    ##from ksi prime to beta vector
    vBeta = (vKsi.T)[:,1:]

    ##covariance matrix of G
    G = np.array(df[factors].drop(columns='C'))
    T = G.shape[0]
    m = G.shape[1]
    
    cov_G = G.T@G/(T-m-1)
    
    ##variance of estimation given the factors
    var_est_fact = vBeta @ cov_G @ vBeta.T + dSigma
    print('Covariance matrix with factor model: \n',var_est_fact)

    
    ##variance of returns
    df_demeaned = np.array(df[asStock] - np.mean(df[asStock]))
    var_est = (df_demeaned.T @ df_demeaned)/(T-1)
    print('Covariance matrix of returns: \n',var_est)
    

    ##initial weights (all equal)
    vW0 = np.ones((1,nStocks))/nStocks
    
    ##constraint function so that weights sum up to 1
    Eq_zero = lambda vW: np.sum(vW)-1
    tCons = ({'type': 'eq','fun': Eq_zero })

    
    ##variance calculation function
    def VarStocks(vW,var):
     return vW @ var @ vW.T
    
    ###optimize weights that minimizes variance given the factors and without the factors known
    res_factor = opt.minimize(VarStocks , vW0 ,args =(var_est_fact) , method ="SLSQP", constraints = tCons)
    res = opt.minimize(VarStocks , vW0 ,args =(var_est) , method ="SLSQP", constraints = tCons)
    
    
    weights = pd.DataFrame(data = {'Weights with Factor Model':res_factor.x,'Weights without Factor Model':res.x},index=asStock)
    print('Optimal weights:\n',weights)
    
###########################################################
### start main
if __name__ == "__main__":
    main()

    