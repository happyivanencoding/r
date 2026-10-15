import copy
import os
import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

from GetBench import Launch_Get_Bench
from notification import sendNotification

# from pandas.errors import SettingWithCopyWarning
# import pickle
# warnings.simplefilter(action='ignore', category=(SettingWithCopyWarning))
warnings.simplefilter(action='ignore', category=(UserWarning))

# Function to Calculate Cumulative Return ---
def calculate_cumulative_return(df, start_date, end_date):
    """
    Calculates the cumulative return for each column in a DataFrame
    between a start and end date.
    """
    # Filter the DataFrame for the specified date range
    period_returns = df.loc[start_date:end_date]

    # Calculate cumulative returns: (1+r1)*(1+r2)*...*(1+rn) - 1
    cumulative_return = (1 + period_returns).prod() - 1
    return cumulative_return


def compute_bench_returns(bench, returns, col_weights='Weight', col_sort='SEDOL'):

    bench = bench[bench[col_sort].notna()]
    list_sedol = list(set(bench[col_sort].values).intersection(set(returns.columns)))
    returns = returns[list_sedol]
    bench = bench[bench[col_sort].isin(list_sedol)]
    returns = returns.loc[:,~returns.columns.duplicated()]
    returns.sort_index(axis=1, inplace=True)
    weights = bench.sort_values(by=col_sort)[col_weights]
    weights_t = np.zeros(shape=(len(returns),len(weights)))
    for i in range(len(returns)-1,-1, -1):
        if i == len(returns)-1:
            weights_t[i] = weights.values
        else:
            weights_t[i] = weights_t[i+1]/(1+(returns.iloc[i,:]).values)
        weights_t[i] = weights_t[i]/weights_t[i].sum()
    return (returns*weights_t).sum(axis=1)

def compute_beta_cov(df, bench_returns, returns, col_sedol='SEDOL', id='ISIN'):

    df = df[df[col_sedol].notna()]
    return df.set_index(id)[col_sedol].apply(lambda x: returns[x].cov(bench_returns)/bench_returns.var())


def compute_te_and_contrib(w_ptf, w_bench, cov, isin_col='ISIN', col_weight='weight', sedol_col='SEDOL'):
    # Merge benchmark weights with portfolio weights on ISIN.
    # We use a left join so that all instruments in w_bench are kept.
    # Any ISIN in w_ptf but not in w_bench will be dropped.
    w_ptf.drop_duplicates(subset=['ISIN'], keep='first',inplace=True)
    w_bench.drop_duplicates(subset=['ISIN'], keep='first',inplace=True)

    merged = w_bench.merge(
        w_ptf[[isin_col, col_weight]],
        on=isin_col,
        how='left',
        suffixes=('', '_ptf')
    )
    
    # If an instrument from w_bench is not in w_ptf, assume portfolio weight is zero.
    merged[col_weight + '_ptf'] = merged[col_weight + '_ptf'].fillna(0)
    
    # Compute active (excess) weight as portfolio weight minus benchmark weight.
    # (This is equivalent to: -[benchmark weight - portfolio weight])
    merged['active_weight'] = merged[col_weight + '_ptf'] - merged[col_weight]
    
    # For TE calculation, we need the covariance matrix to align with our instruments.
    # Here we assume that w_bench (and therefore merged) has a sedol column
    # that can be used to index the covariance matrix.
    # Set the index of merged to the sedol identifier.
    merged = merged.set_index(sedol_col)
    
    # Ensure we only use instruments that exist in the covariance matrix.
    merged = merged.loc[merged.index.intersection(cov.index)]
    
    # Extract the active weight vector in the order of merged's index.
    active_weight = merged['active_weight'].values
    
    # Subset the covariance matrix so that its rows/columns match the order of merged.
    cov_sub = cov.loc[merged.index, merged.index].values
    
    # Compute tracking error (TE)
    te = np.sqrt(np.dot(active_weight.T, np.dot(cov_sub, active_weight)))
    
    # Compute the marginal contribution for each instrument:
    # For each instrument i, the contribution is given by:
    #   contrib_i = ( (cov_row_i dot active_weight) * active_weight_i ) / te
    # where cov_row_i is the i-th row of the covariance matrix (restricted to merged index).
    contrib_te = pd.Series(index=merged.index, dtype=float)
    for inst in merged.index:
        cov_row = cov.loc[inst, merged.index].values
        weight_i = merged.loc[inst, 'active_weight']
        contrib_te.loc[inst] = (np.dot(cov_row, active_weight) * weight_i) / te if te != 0 else 0
    contrib_te=pd.DataFrame(contrib_te).reset_index(drop=False)
    contrib_te.rename(columns={'index':'Company SEDOL'},inplace=True)
    merged=merged.reset_index(drop=False)
    merged.rename(columns={'index':'Company SEDOL'},inplace=True)
    contrib_te=contrib_te.merge(how='left', right=merged[['Company SEDOL','ISIN']],left_on="Company SEDOL",right_on="Company SEDOL")
   

    contrib_te.set_index("ISIN",inplace=True)
    contrib_te=contrib_te.drop(columns=["Company SEDOL"])
    return te, contrib_te


def update_beta_te_vol(ptf_modele, ind, returns, freq=252, col_isin='ISIN', col_sedol='SEDOL', col_weight='weight', col_weight_ptf='Weight PTF Modele', only_beta_vol=False):

    ptf=ptf_modele.rename(columns={col_weight_ptf:col_weight})
    ptf[col_weight] = ptf[col_weight].fillna(0)
    ind[col_weight] = ind[col_weight].fillna(0)

    ptf=ptf[ptf[col_sedol].isin(returns.columns)]
    ind=ind[ind[col_sedol].isin(returns.columns)]

    ptf[col_weight] = ptf[col_weight]/ptf[col_weight].sum()
    ind[col_weight] = ind[col_weight]/ind[col_weight].sum()
    ptf = ptf[ind.columns]

    ptf_wo_na = ptf[ptf[col_sedol].notna()]
    ind_wo_na = ind[ind[col_sedol].notna()]

    bench_returns = compute_bench_returns(ind,returns,col_weight,col_sedol)
    vol_bench = bench_returns.std()*np.sqrt(freq)

    # betas = compute_beta_cov(ptf, bench_returns,returns,col_sedol) # OLD METHODO

    # NEW METHODO ####################################################################################################
    market_df=ind
    market_df = market_df.dropna(subset=[col_sedol])
    market_df[col_weight] /= market_df[col_weight].sum()
    market_weights = market_df.set_index(col_sedol)[col_weight]
    market_returns = returns[market_df[col_sedol]]
    market_returns_agg = market_returns.dot(market_weights)
    market_returns_agg.name = 'market_return'
    market_variance = market_returns_agg.var()

    ptf.dropna(subset=["Company SEDOL"],inplace=True)
    ptf[col_weight] /= ptf[col_weight].sum()

    sedol_in_portfolio = ptf[col_sedol].unique()
    portfolio_security_returns = returns[sedol_in_portfolio]

    cov_with_market = portfolio_security_returns.apply(lambda x: x.cov(market_returns_agg))
    security_betas = cov_with_market / market_variance
    security_betas.name = 'beta'

    ptf = ptf.merge(security_betas, how='left', left_on=col_sedol,right_index=True)
    betas=ptf[['beta','ISIN']]
    betas.drop_duplicates(subset=['ISIN'], keep='first',inplace=True)
    betas=betas.set_index('ISIN',drop=True)
    ##############################################################################################################################




    vol = ptf_wo_na.set_index(col_isin)[col_sedol].apply(lambda x: returns[x].std()*np.sqrt(freq))
    if only_beta_vol:
        return betas, vol
    missing_in_univ = list(set(ptf_wo_na[col_isin]) - set(ind_wo_na[col_isin]))
    if len(missing_in_univ)>0:
        add_to_univ = ptf_wo_na[ptf_wo_na[col_isin].isin(missing_in_univ)]
        add_to_univ['Weight'] = 0
        univ = pd.concat([ind_wo_na,add_to_univ])
    else:
        univ=copy.deepcopy(ind_wo_na)
    univ = univ[univ[col_sedol].isin(returns.columns)] 
    ptf_wo_na = ptf_wo_na[ptf_wo_na[col_sedol].isin(returns.columns)] 
    cov_univ = returns[univ[col_sedol]].cov()*freq
    cov_ptf = returns[ptf_wo_na[col_sedol]].cov()*freq
    weight_ptf = ptf_wo_na[[col_sedol,col_isin,col_weight]]
    weight_univ = univ[[col_sedol,col_isin,col_weight]]

    cov_univ = cov_univ.loc[:, ~cov_univ.columns.duplicated()]
    cov_ptf = cov_ptf.loc[:, ~cov_ptf.columns.duplicated()]
    cov_univ = cov_univ[~cov_univ.index.duplicated(keep="first")]
    cov_ptf = cov_ptf[~cov_ptf.index.duplicated(keep="first")]
    weight_ptf.drop_duplicates(keep='first',inplace=True)
    weight_univ.drop_duplicates(keep='first',inplace=True)

    te, contrib_te = compute_te_and_contrib(weight_ptf, weight_univ, cov_univ, col_isin, col_weight, col_sedol)
    vol_ptf = np.sqrt(weight_ptf[col_weight].values.transpose() @ cov_ptf.values @ weight_ptf[col_weight].values)
    return betas, te, contrib_te, vol_ptf, vol,vol_bench



def read_fonds(filename):
    
    xl = pd.ExcelFile(filename)
    list_sheet = xl.sheet_names
    dict_fund = {}
    dict_bench = {}
    for sheet in list_sheet:
        if sheet =='Mapping fonds indice':
            mapping_fund_ind = xl.parse(sheet, header=0)
        else:
            if sheet in list(mapping_fund_ind['Fonds'].values):
                dict_fund[sheet]=xl.parse(sheet, header=0)
            elif sheet in list(mapping_fund_ind['Indice'].values):
                dict_bench[sheet]=xl.parse(sheet, header=0)
    return dict_fund, dict_bench, mapping_fund_ind




def add_metrics_to_fund(fund_name,ind_name,fund, screen, returns, indice, capital_iq, equipe, sector_col):

    

    if fund_name=="CM-AM Alizes Tempere":
        exclude_list = ['NOK','USD','DKK','GBP','EUR','CHF','SEK','JPY','CNY','KRY','SGD','CAD','AUD','ZAR']
        fund = fund[~fund['ISIN'].isin(exclude_list)]

    #Enlever les trackers
    if (equipe =="Epargne_Salariale") | (fund_name == "CM-AM SUSTAINABLE PLANET"):
        fund = fund[(fund["typeAction"]!="tracker") & (fund["classeActif"]=="Actions")]
        print("Equipe Epargne Salariale")

    # fund = fund[(fund["typeAction"]!="tracker")]
    fund = fund[['ISIN','LIBELLE','%ACTIF']]

    #  
    fund['ISIN'] = fund['ISIN'].str.split(' ').str[0]
    fund['%ACTIF 100%'] = fund['%ACTIF']/fund['%ACTIF'].sum()

    indice['%ACTIF']=indice['%ACTIF'].replace({'^([A-Za-z]|[0-9]|_)+$': 0}, regex=True)
    indice['%ACTIF'] =pd.to_numeric(indice['%ACTIF'], errors ='coerce').fillna(0).astype('float')
    
    indice['%ACTIF 100%'] = indice['%ACTIF']/indice['%ACTIF'].sum()

    fund_full = fund.merge(how='left', right= screen[['ISIN', sector_col, 'Zone Geo' ,'Exchange Country Name', 'Benchmark Market Value Millions in EUR', 'Dividend Avg Percentile','Value Avg Percentile','Quality Avg Percentile','Mom Avg Percentile','LowVol Avg Percentile','Growth Avg Percentile']], on='ISIN')
    fund_full = fund_full.merge(how='left', right=capital_iq[['ISIN', 'Reco Analyst']], on = 'ISIN')
    
    
    indice_full = indice.merge(how='left', right= screen[['ISIN', sector_col, 'Zone Geo' , 'Exchange Country Name', 'Benchmark Market Value Millions in EUR','Dividend Avg Percentile','Value Avg Percentile','Quality Avg Percentile','Mom Avg Percentile','LowVol Avg Percentile','Growth Avg Percentile']], on='ISIN')
    indice_full = indice_full.merge(how='left', right=capital_iq[['ISIN', 'Reco Analyst']], on = 'ISIN')


    bench = screen.merge(how='left', right=indice_full[['ISIN', '%ACTIF']], on='ISIN')
    bench.rename(columns={'%ACTIF':'weight'}, inplace=True)
    bench=bench[bench['weight']>0]
    
    ptf = bench.merge(how='outer', right = fund[['ISIN','%ACTIF']], on = 'ISIN')
    ptf = ptf.drop('Company SEDOL', axis=1)
    ptf = ptf.merge(how='left', right=screen[['ISIN', 'Company SEDOL']], on='ISIN')
    ptf['%ACTIF'] = ptf['%ACTIF'].fillna(0)
    # A=ptf.head()
    bench['Weight']=bench['weight']
    
    betas, te_ptf, contrib_te_ptf, vol_ptf, vol, vol_bench = update_beta_te_vol(ptf, bench, returns, 252, 'ISIN', 'Company SEDOL', 'Weight', '%ACTIF')
    
    fund_full.drop_duplicates(subset=['ISIN'], keep='first',inplace=True)
    indice_full.drop_duplicates(subset=['ISIN'], keep='first',inplace=True)
    fund_full.set_index('ISIN', inplace=True)
    indice_full.set_index('ISIN', inplace=True)

    fund_full['Beta'] = betas
    fund_full['Contrib TE'] = contrib_te_ptf
    indice_full['Beta'] = betas
    indice_full['Contrib TE'] = 0

    fund_full.reset_index(inplace=True)
    indice_full.reset_index(inplace=True)
    fund_full['Dividend Avg Percentile'] = fund_full['Dividend Avg Percentile'].fillna(0)
    indice_full['Dividend Avg Percentile'] = indice_full['Dividend Avg Percentile'].fillna(0)

    hors_indice_id = list(set(fund_full['ISIN']) - set(indice_full['ISIN']))
    fund_full['Hors indice'] = 0
    indice_full['Hors indice'] = 0
    
    fund_full.loc[fund_full['ISIN'].isin(hors_indice_id), 'Hors indice'] = 1
    # hors_indice_df = fund_full.loc[fund_full['Hors indice'] == 1, fund_full.columns[0:3]]
    #ajouté
    hors_indice_df = fund_full.loc[fund_full['Hors indice'] == 1][['ISIN', "LIBELLE", '%ACTIF' ]]
    #


    # hors_indice_df = pd.concat([hors_indice_df,pd.DataFrame(columns=hors_indice_df.columns, data=[[float('NaN')]*len(hors_indice_df.columns)]*100)], axis=0)
    # fund_full = pd.concat([fund_full,pd.DataFrame(columns=fund_full.columns, data=[[float('NaN')]*len(fund_full.columns)]*500)], axis=0)
    # indice_full = pd.concat([indice_full,pd.DataFrame(columns=indice_full.columns, data=[[float('NaN')]*len(indice_full.columns)]*2000)], axis=0)
    
    fund_full = fund_full.merge(how='left', right=capital_iq[['ISIN','Score ML','Daily Vol 90J',
         'EBITDAm FY1','Gross Margin','Earns Yield FY1','DVD Payout FY0','DVD Yield FY1','FCF Conversion','CFO Div Cov Ratio',
        'FCF Div Cov Ratio','PE FY1','Price to Book FY1','EV to Sales FY1','EV To EBITDA FY1','EV to Ebit FY1', 'Sales Growth FY1',
        'EBITDA Growth FY1 CIQ','Gross Income Growth FY1','EPS Growth FY1','Sales_5Y_growth','EPS_5Y_growth','EPS Revision Ratio',
        'Pct_Short_Interest',
        'SP Price Target CIQ']], on = 'ISIN')
    # 'Number Pos. Surprise Earnings (last 8Q)','Number Pos. Surprise Sales (last 8Q)',
    
    indice_full = indice_full.merge(how='left', right=capital_iq[['ISIN','Score ML','Daily Vol 90J',
         'EBITDAm FY1','Gross Margin','Earns Yield FY1','DVD Payout FY0','DVD Yield FY1','FCF Conversion','CFO Div Cov Ratio',
        'FCF Div Cov Ratio','PE FY1','Price to Book FY1','EV to Sales FY1','EV To EBITDA FY1','EV to Ebit FY1', 'Sales Growth FY1',
        'EBITDA Growth FY1 CIQ','Gross Income Growth FY1','EPS Growth FY1','Sales_5Y_growth','EPS_5Y_growth','EPS Revision Ratio',
        'Pct_Short_Interest',
        'SP Price Target CIQ']], on = 'ISIN')
    # 'Number Pos. Surprise Earnings (last 8Q)','Number Pos. Surprise Sales (last 8Q)',
    
    # hors_indice_df = pd.concat([hors_indice_df,pd.DataFrame(columns=hors_indice_df.columns, data=[[float('NaN')]*len(hors_indice_df.columns)]*100)], axis=0)
    # fund_full = pd.concat([fund_full,pd.DataFrame(columns=fund_full.columns, data=[[float('NaN')]*len(fund_full.columns)]*500)], axis=0)
    # indice_full = pd.concat([indice_full,pd.DataFrame(columns=indice_full.columns, data=[[float('NaN')]*len(indice_full.columns)]*2000)], axis=0)
    
    fund_full['EPS Growth FY1'] = fund_full['EPS Growth FY1'].clip(lower=-100, upper=1000)
    indice_full['EPS Growth FY1'] = indice_full['EPS Growth FY1'].clip(lower=-100, upper=1000)
    fund_full['FCF Conversion'] = fund_full['FCF Conversion'].clip(lower=-100, upper=1000)
    indice_full['FCF Conversion'] = indice_full['FCF Conversion'].clip(lower=-100, upper=1000)
    fund_full['EBITDAm FY1'] = fund_full['EBITDAm FY1'].clip(lower=-100, upper=100)
    indice_full['EBITDAm FY1'] = indice_full['EBITDAm FY1'].clip(lower=-100, upper=100)


    fund_full=fund_full.sort_values(by=['%ACTIF'], ascending=False)
    indice_full=indice_full.sort_values(by=['%ACTIF'], ascending=False)
    

    
    return fund_full, indice_full, te_ptf, hors_indice_df






def read_fonds_server(filename):

    mynewlist=filename
    dict_fund2 = {}
    dict_bench2 = {}
    mapping_fund_ind2 =pd.DataFrame(columns=['Fonds','Indice'])
    for e in mynewlist:
        ligneListe=e
        NamePtf=ligneListe[0]
        Ptf_=pd.DataFrame(ligneListe[1])
        # Ptf_=Ptf_[['isin','nom','percentActif']]
        Ptf_.rename(columns={'isin':'ISIN','nom':'LIBELLE','percentActif':'%ACTIF'},inplace=True)

        NameBench=ligneListe[2]
        Bench_=ligneListe[3]
        Bench_=Bench_[['MAIN_SECURITY_CODE','SECURITY_NAME','SECURITY_WEIGHT']]
        Bench_.rename(columns={'MAIN_SECURITY_CODE':'ISIN','SECURITY_NAME':'LIBELLE','SECURITY_WEIGHT':'%ACTIF'},inplace=True)


        dict_fund2[NamePtf]=Ptf_
        dict_bench2[NameBench]=Bench_

        mapping_ =pd.DataFrame({'Fonds': [NamePtf], 'Indice': [NameBench]})
        mapping_fund_ind2=pd.concat([mapping_fund_ind2,mapping_],axis=0)
        
    return dict_fund2, dict_bench2, mapping_fund_ind2


# Fonction pour classer les colonnes par secteur avec proportions et transformer en scores de 0 à 10
def normalize_by_sector(df, colonnes, secteur_colonne):
    df_transforme = df.copy()
    for secteur in df[secteur_colonne].unique():
        secteur_df = df[df[secteur_colonne] == secteur]
        for colonne in colonnes:
            # Calculer les rangs sous forme de proportions
            rangs_proportions = secteur_df[colonne].rank(pct=True)
            # Transformer les proportions en scores de 0 à 10
            scores = rangs_proportions * 10
            df_transforme.loc[df[secteur_colonne] == secteur, colonne] = scores
    return df_transforme



                


                

# LAUNCHER
icb20_to_icb19 = {
    "Automobiles and Parts": 1,
    "Banks": 2,
    "Basic Resources": 3,
    "Chemicals": 4,
    "Construction and Materials": 5,
    "Financial Services": 6,
    "Food, Beverage and Tobacco": 7,
    "Health Care": 8,
    "Industrial Goods and Services": 9,
    "Insurance": 10,
    "Media": 11,
    "Energy": 12,
    "Consumer Products and Services": 13,
    "Real Estate": 14,
    "Retail": 15,
    "Personal Care, Drug and Grocery Stores": 15,
    "Technology": 16,
    "Telecommunications": 17,
    "Travel and Leisure": 18,
    "Utilities": 19,
}

transpa_icb19 = pd.Series(data = ['Automobiles and Parts','Banks','Basic Resources','Chemicals','Construction and Materials','Financial Services','Food, Beverage and Tobacco','Health Care','Industrial Goods and Services',
                                'Insurance','Media','Energy','Personal and Household Goods','Real Estate','Retail','Technology','Telecommunications','Travel and Leisure','Utilities'],
                                index= [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], name='transpa_icb19')


def generate_report(sector_col):
    today_str = datetime.today().strftime('%Y%m%d')
    date_folder = datetime.today().strftime("%Y%m")

    file_fund = Launch_Get_Bench()

    msg=''

    if sys.platform.startswith('win'):
        wb_input_path=r"\\groupe-ufg.com\Commun\Prive\GestionAM\Ingenierie_Financiere\PROD\_BASE\_FONDS_CMAM_GET_LT\X\TBB_Gestion_Action.xlsx"
        wb_output_folder_path=r"\\groupe-ufg.com\Commun\Prive\GestionAM\Ingenierie_Financiere\PROD\_BASE\_FONDS_CMAM_GET_LT\X"
        file_returns=r"\\groupe-ufg.com\commun\Prive\GestionAM\Ingenierie_Financiere\PROD\_EQUITY\0_RETURNS\returns_2Y.parquet"
        path_ciq=r"\\groupe-ufg.com\commun\Prive\GestionAM\Ingenierie_Financiere\PROD\_EQUITY\0_SCREEN_AGG\last_screenCIQ.parquet"
        output_path_equipe_caroline=r"\\groupe-ufg.com\Commun\Prive\GestionAM\Gestion Actions\_INGEFI\POSITIONS_FONDS_INDICES"
        output_path_equipe_ES =r"\\groupe-ufg.com\Commun\Prive\GestionAM\Ingenierie_Financiere\0_INGE_FI\Epargne_Salariale\Report_funds_equity"
        Path_news=r"\\groupe-ufg.com\Commun\Prive\GestionAM\Ingenierie_Financiere\PROD\MAJ_news_factset_daily\0_DATA\Base_news_facset_BRUTE.parquet"
        path_screen_hist = r"\\groupe-ufg.com\Commun\Prive\GestionAM\Ingenierie_Financiere\PROD\_EQUITY\0_SCREEN_AGG\screen_aggregate.parquet"
        path_reco_facto =  r"\\groupe-ufg.com\Commun\Prive\GestionAM\Ingenierie_Financiere\PROD\_EQUITY\0_PTF_BLOOM\reco_secto_facto.xlsx"
        path_masque=r"\\groupe-ufg.com\Commun\Prive\GestionAM\Ingenierie_Financiere\PROD\_BASE\_FONDS_CMAM_GET_LT\MASQUE_GET_BENCH_AND_EQUITY_FUND.xlsx"
        path_exp_ml_world = fr"\\groupe-ufg.com\Commun\Prive\GestionAM\Ingenierie_Financiere\PROD\_EQUITY\1_FACTEUR_ML\1_PROD_OPTIM\Input_Output\{date_folder}\Output_SUIVI\ML_Explicativite_WORLD.xlsx"
    

    else :  
        wb_input_path=r'/usr/share/inge-fi/PROD/_BASE/_FONDS_CMAM_GET_LT/X/TBB_Gestion_Action.xlsx'
        wb_output_folder_path=r'/usr/share/inge-fi/PROD/_BASE/_FONDS_CMAM_GET_LT/X'
        file_returns=r'/usr/share/inge-fi/PROD/_EQUITY/0_RETURNS/returns_2Y.parquet'
        path_ciq=r'/usr/share/inge-fi/PROD/_EQUITY/0_SCREEN_AGG/last_screenCIQ.parquet'
        output_path_equipe_caroline=r'/usr/share/gestionactions/POSITIONS_FONDS_INDICES'
        output_path_equipe_ES =r"/usr/share/inge-fi/0_INGE_FI/Epargne_Salariale/Report_funds_equity"
        Path_news=r'/usr/share/inge-fi/PROD/MAJ_news_factset_daily/0_DATA/Base_news_facset_BRUTE.parquet'
        path_screen_hist = r"/usr/share/inge-fi/PROD/_EQUITY/0_SCREEN_AGG/screen_aggregate.parquet"
        path_reco_facto =  r"/usr/share/inge-fi/PROD/_EQUITY/0_PTF_BLOOM/reco_secto_facto.xlsx"
        # path_folder=r'/usr/share/rimes/'
        path_masque=r'/usr/share/inge-fi/PROD/_BASE/_FONDS_CMAM_GET_LT/MASQUE_GET_BENCH_AND_EQUITY_FUND.xlsx'
        path_exp_ml_world = fr"/usr/share/inge-fi/PROD/_EQUITY/1_FACTEUR_ML/1_PROD_OPTIM/Input_Output/{date_folder}/Output_SUIVI/ML_Explicativite_WORLD.xlsx"


    # LOAD SCREEN AGG et RECO FACTO SECTO
    try :
        df_reco_facto = pd.read_excel(path_reco_facto,sheet_name = "facto_eu", index_col=0)

        df_masque = pd.read_excel(path_masque)
        df_masque_short = df_masque[["Nom Fichier pickle", 'Equipe']]

        # Conserver les treize dernières dates pour calculer les évolutions.
        dates = pd.to_datetime(pd.read_parquet(path_screen_hist, columns=["Date"])["Date"])
        dates = dates.dropna().drop_duplicates().sort_values().tolist()
        if len(dates) < 13:
            raise ValueError("Historique insuffisant : treize dates sont nécessaires.")

        spot_date = dates[-1]
        date_1M_ago = dates[-2]
        date_3M_ago = dates[-4]
        date_6M_ago = dates[-7]
        date_12M_ago = dates[-13]

        screen_agg = pd.read_parquet(
            path_screen_hist, filters=[("Date", "in", dates[-13:])]
        ).reset_index()
        if sector_col not in screen_agg.columns:
            if sector_col == "ICB20" and "ICB20 Supersector" in screen_agg.columns:
                screen_agg["ICB20"] = screen_agg["ICB20 Supersector"]
            else:
                raise ValueError(f"Colonne de secteur absente : {sector_col}")
        if sector_col == " Benchmark ICB Supersector ":
            # Convertir les secteurs ICB20 en codes ICB19 historiques.
            icb20_col = "ICB20" if "ICB20" in screen_agg.columns else "ICB20 Supersector"
            secteurs_icb20 = screen_agg[icb20_col].replace(
                [0, "0", "0.0", "", "_", "@NA"], np.nan
            ).astype("string").str.strip()
            secteurs_inconnus = sorted(set(secteurs_icb20.dropna()) - set(icb20_to_icb19))
            if secteurs_inconnus:
                raise ValueError(f"Secteurs ICB20 inconnus : {secteurs_inconnus}")
            screen_agg[sector_col] = secteurs_icb20.map(icb20_to_icb19)
        screen_agg[sector_col] = screen_agg[sector_col].replace(
            [0, "0", "0.0", ""], np.nan
        )

        last_screen = screen_agg.loc[screen_agg["Date"] == spot_date].copy()
        last_screen = last_screen.drop_duplicates(subset="ISIN")
        if last_screen[sector_col].notna().sum() == 0:
            raise ValueError(f"Aucun secteur renseigné dans {sector_col} au {spot_date.date()}.")
        if sector_col == " Benchmark ICB Supersector ":
            last_screen[sector_col] = last_screen[sector_col].map(transpa_icb19)
            print(f"Couverture ICB20 : {last_screen[sector_col].notna().sum()} / {len(last_screen)}")
        last_screen["Zone Geo"] = last_screen["Exchange Country Region"].replace(
            ["@NA", "Pacific", "Mid East", "South America", "0", "East Europe", "Africa"],
            "Others"
        )
        # Reprendre le secteur actuel pour comparer les scores historiques.
        secteurs = screen_agg.loc[screen_agg["Date"] == spot_date].drop_duplicates("ISIN").set_index("ISIN")[sector_col]
        screen_agg[sector_col] = screen_agg["ISIN"].map(secteurs).combine_first(
            screen_agg[sector_col]
        )
        screen_agg_filtered = screen_agg

        # Compléter les capitalisations absentes.
        cap = "Benchmark Market Value Millions in EUR"
        last_screen[cap] = last_screen[cap].fillna(
            last_screen["Weight in MSCI ACWI"] * 600000
        ).fillna(last_screen["Weight in MSCI EUR SMALL"] * 12000)

        dict_facteurs = {'Value Avg Percentile' : "Score Value",
                                                        "Growth Avg Percentile" : "Score Growth",
                                                        "Quality Avg Percentile" : "Score Quality",
                                                        "Mom Avg Percentile" : "Score Momentum",
                                                        "LowVol Avg Percentile" : "Score Volatility",
                                                        "Multi Avg Percentile" : "Score Multifacteur"}

        screen_agg_filtered = screen_agg_filtered.rename(columns = dict_facteurs)   
        msg = msg + f"OK OPEN screenAgg ({spot_date.date()}, {sector_col})"
    except Exception as e:
        sendNotification("jobprod",  "Generate Report Equity : ", "KO OPEN screenAgg : "  + " // " + str(e))
        print("Generate Report Equity : ", "KO OPEN : screenAgg, reco Facto",e)
        raise

    #############################################################################################################################################
    # LECTURE POS FUND AND BENCH
    try:
        dict_fund2, dict_bench2, mapping_fund_ind2 = read_fonds_server(file_fund)
        mapping_fund_ind2 = dict(zip(mapping_fund_ind2['Fonds'], mapping_fund_ind2['Indice']))
        print("Base pos open")
        # sendNotification("jobprod",  "Generate Report Equity : ", "OK OPEN : " + file_fund)
        msg=msg + "\n" + "OK OPEN : " + "file_fund"
        print("file_fund","\n")
        print("Generate Report Equity : ", "OK OPEN : " + "file_fund")
    except Exception as e:
        sendNotification("jobprod",  "Generate Report Equity : ", "KO OPEN : " + "file_fund" + " // " + str(e))
        print("Generate Report Equity : ", "KO OPEN : " + "file_fund",e)
    time.sleep(10)
    #############################################################################################################################################
    # LECTURE RETURNS
    try:
        df_returns = pd.read_parquet(file_returns)
        df_returns = df_returns[df_returns.index >=df_returns.index[-1] - pd.Timedelta(days=365) ]
        print("Base return open")
        print(file_returns,"\n")
        # sendNotification("jobprod",  "Generate Report Equity : ", "OK OPEN : " + file_returns)
        msg=msg + "\n" + "OK OPEN : " + file_returns
        print("Generate Report Equity : ", "OK OPEN : " + file_returns)
    except Exception as e:
        sendNotification("jobprod",  "Generate Report Equity : ", "KO OPEN : " + file_returns + " // " + str(e))
        print("Generate Report Equity : ", "KO OPEN : " + file_returns)
    time.sleep(10)
    #############################################################################################################################################
    # CIQ
    try:
        # screen_ciq = pd.read_excel(path_ciq,header = 0, na_values=["@NA", "#N/A"])
        screen_ciq = pd.read_parquet(path_ciq)
        screen_ciq=screen_ciq.reset_index(inplace=False)
        screen_ciq['Reco Analyst']= screen_ciq['Reco Analyst'].astype(str)

        column_to_delete = 'Unnamed: 0'
        if column_to_delete in screen_ciq.columns:
            screen_ciq.drop(columns=[column_to_delete], inplace=True)
        print("Base CIQ open")
        print(path_ciq,"\n")

        screen_ciq['Reco Analyst'] = screen_ciq['Reco Analyst'].str.split('(',expand=True)[0]
        screen_ciq['Reco Analyst'] = screen_ciq['Reco Analyst'].str.strip()
        screen_ciq = screen_ciq[~screen_ciq['Company SEDOL'].duplicated(keep='first')]
        screen_ciq = screen_ciq[~screen_ciq['ISIN'].duplicated(keep='first')]
        # sendNotification("jobprod",  "Generate Report Equity : ", "OK OPEN : " + path_ciq)
        msg=msg + "\n" + "OK OPEN : " + path_ciq
        print("Generate Report Equity : ", "OK OPEN : " + path_ciq)
    except Exception as e:
        sendNotification("jobprod",  "Generate Report Equity : ", "KO OPEN : " + path_ciq + " // " + str(e))
        print("Generate Report Equity : ", "KO OPEN : " + path_ciq,e)
    time.sleep(10)

    sendNotification("jobprod",  "Generate Report Equity : ", "Preprocess OK : " + "\n" + msg)

    list_ind_updated = []

    msg2=''
    for fund in dict_fund2:
        try:
            equipe = df_masque_short[df_masque_short["Nom Fichier pickle"]==fund].values[0,1]
            # if fund=="CM-AM Alizes Tempere":
            if fund!="Essai":

                print(fund + " // " + mapping_fund_ind2[fund])
                try:
                    fund_full, indice_full, te_ptf, hors_indice_df = add_metrics_to_fund(fund,mapping_fund_ind2[fund],dict_fund2[fund], last_screen, df_returns, dict_bench2[mapping_fund_ind2[fund]], screen_ciq,equipe,sector_col)
                except Exception as e:
                    sendNotification("jobprod",  "Generate Report Equity : ", "add_metrics_to_fund KO : " + fund + str(e))


                # FEUILLE MASQUE DES ANALYSES
                try:
                    wb = load_workbook(wb_input_path)  # load as openpyxl workbook; useful to keep the original layout
                    # Mettre à jour les références au secteur dans les formules Excel.
                    for feuille in wb:
                        for cellule in feuille._cells.values():
                            if cellule.data_type == "f" and isinstance(cellule.value, str):
                                cellule.value = cellule.value.replace(
                                    " Benchmark ICB Supersector ", sector_col
                                )
                except Exception as e:
                    sendNotification("jobprod",  "Generate Report Equity : ", "load wb KO : " + fund + str(e))

                # FEUILLE DATA
                try :
                    # Étape 1 : Extraire les valeurs uniques de la colonne 'ISIN'
                    isin_fund_unique = fund_full['ISIN'].unique()
                    isin_indice_unique = indice_full['ISIN'].unique()

                    # Étape 2 : Fusionner les deux ensembles d'ISINs uniques
                    all_isin_unique = pd.Series(pd.concat([pd.Series(isin_fund_unique), pd.Series(isin_indice_unique)])).unique()


                    screen_agg_filtered_fund = screen_agg_filtered[screen_agg_filtered["ISIN"].isin(all_isin_unique)].drop_duplicates()

                    screen_spot = screen_agg_filtered_fund[screen_agg_filtered_fund["Date"]==spot_date]
                    screen_1M_ago =  screen_agg_filtered_fund[screen_agg_filtered_fund["Date"]==date_1M_ago]
                    screen_3M_ago =  screen_agg_filtered_fund[screen_agg_filtered_fund["Date"]==date_3M_ago]
                    screen_6M_ago =  screen_agg_filtered_fund[screen_agg_filtered_fund["Date"]==date_6M_ago]
                    screen_12M_ago =  screen_agg_filtered_fund[screen_agg_filtered_fund["Date"]==date_12M_ago]

                    columns_to_norm = list(dict_facteurs.values())

                    sector_column = sector_col

                    screen_1M_ago_norm = normalize_by_sector(screen_1M_ago, columns_to_norm, sector_column )
                    screen_3M_ago_norm = normalize_by_sector(screen_3M_ago, columns_to_norm, sector_column )
                    screen_6M_ago_norm = normalize_by_sector(screen_6M_ago, columns_to_norm, sector_column )
                    screen_12M_ago_norm = normalize_by_sector(screen_12M_ago, columns_to_norm, sector_column )
                    screen_spot_norm = normalize_by_sector(screen_spot, columns_to_norm  + ["Score ML"], sector_column )


                    value_columns = ["Score Value" , 
                                    'PE FY1', 
                                    'Price to Book FY1', 
                                    'Price to FreeCF FY1', 
                                    'EV to Ebit FY1', 
                                    'EV to Sales FY1', 
                                    'EV To EBITDA FY1', 
                                    'PE LTM', 
                                    'PB LTM', 
                                    'PFCF LTM', 
                                    "EV to Sales LTM",
                                    "EV To EBITDA LTM" ]

                    growth_columns = ["Score Growth",
                                    "5Y_Hist EPS TrendStab",
                                    "5Y_Hist GrossInc TrendStab",
                                    "5Y_Hist Sales TrendStab"]

                    quality_columns = ["Score Quality",
                                    "ROE avg FY0",
                                    "Oper Margin",
                                    "Asset TO exFIN",
                                    "NetDebt to EBITDA exFIN",
                                    "TIER1 Ratio FY0",
                                    "ROTE avg FY1",
                                    "Combined Ratio FY1" 
                                    ]

                    momentum_columns = ["Score Momentum",
                                        "PMOM 12M1M",
                                        "EPS NTM 3M Growth",
                                        "EPS Revision Ratio"]

                    lowvol_columns = ["Score Volatility",
                                    "Daily Vol 60J",
                                    "Daily Vol 90J",
                                    "Daily Vol 260J"]

                    multifactor_score = ["Score Multifacteur"]

                    columns_to_compute = multifactor_score + value_columns + growth_columns + quality_columns + momentum_columns + lowvol_columns 
                    columns_classique = ["ISIN", "Name", sector_col]
                    all_columns = columns_classique + columns_to_compute

                    icb_supersectors_mapping = {  
                                1: "Auto & Parts", 
                                2: "Banks", 
                                3: "Basic Resources", 
                                4: "Chemicals",
                                5: "Construction", 
                                6: "Financial Services", 
                                7: "Food, Beverage & Tobacco",
                                8: "Health Care", 
                                9: "Industrial Goods & Services", 
                                10: "Insurance",
                                11: "Media", 
                                12: "Energy", 
                                13: "Personal & Household Goods",
                                14: "Real Estate", 
                                15: "Retail", 
                                16: "Technology",
                                17: "Telecommunications", 
                                18: "Travel & Leisure", 
                                19: "Utilities"
                            }

                    screen_final = pd.merge(screen_spot_norm, screen_1M_ago_norm, on = "ISIN", how = "left", suffixes = ("", "_1M"))
                    screen_final = pd.merge(screen_final, screen_3M_ago_norm, on = "ISIN", how = "left", suffixes = ("", "_3M"))
                    screen_final = pd.merge(screen_final, screen_6M_ago_norm, on = "ISIN", how = "left", suffixes = ("", "_6M"))
                    screen_final = pd.merge(screen_final, screen_12M_ago_norm, on = "ISIN", how = "left", suffixes = ("", "_12M"))
                    order_columns = []


                    order_columns = []
                    for x in columns_to_compute :
                        order_columns += [x, x + "_1M", x + "_DELTA_1M"] 
                        screen_final[x + "_DELTA_1M"] = screen_final[x] - screen_final[x+"_1M"]

                        order_columns += [x + "_3M", x + "_DELTA_3M"] 
                        screen_final[x + "_DELTA_3M"] = screen_final[x] - screen_final[x+"_3M"]

                        order_columns += [x + "_6M", x + "_DELTA_6M"] 
                        screen_final[x + "_DELTA_6M"] = screen_final[x] - screen_final[x+"_6M"]

                        order_columns += [x + "_12M", x + "_DELTA_12M"] 
                        screen_final[x + "_DELTA_12M"] = screen_final[x] - screen_final[x+"_12M"]

                    order_columns = columns_classique + order_columns + ["Score ML"]
                    screen_final = screen_final[order_columns]
                    if sector_col == " Benchmark ICB Supersector ":
                        screen_final[sector_col] = screen_final[sector_col].map(icb_supersectors_mapping)

                    df_reco_facto = df_reco_facto.rename(columns =dict_facteurs)
                    spot_date_bis = spot_date + pd.offsets.MonthBegin(1)  
                    df_reco_facto_spot = df_reco_facto[df_reco_facto.index==spot_date_bis]
                    screen_final.index = screen_final["ISIN"]
                    score_multifacteur_tilt = (df_reco_facto_spot.iloc[0]*screen_final[df_reco_facto_spot.columns]).sum(axis = 1)
                    score_multifacteur_tilt = score_multifacteur_tilt.reset_index().rename(columns = { 0 : "Score Multiffacteur Tilt"})
                    screen_final = screen_final.reset_index(drop = True)
                    screen_final = pd.merge(screen_final, score_multifacteur_tilt, on = "ISIN", how = "left")
                    screen_final = normalize_by_sector(screen_final, ["Score Multiffacteur Tilt"], sector_column )
                    colonne_order = [col for col in screen_final.columns[:4]] + ["Score Multiffacteur Tilt"] + [col for col in screen_final.columns[4:-1]] 
                    screen_final = screen_final[colonne_order]
                    screen_final = screen_final.loc[:, ~screen_final.columns.duplicated()]
                    screen_final = screen_final.sort_values(by=[sector_col, "Score Multiffacteur Tilt"], ascending=[True, False])
                    screen_final = screen_final.rename(columns = {"Score ML" : "Score ML rebased"})
                    screen_final["Score ML rebased"] = screen_final["Score ML rebased"].fillna(-1)


                except Exception as e:
                    sendNotification("jobprod",  "Generate Report Equity : ", "load wb KO : " + fund + str(e))
                    print("Generate Report Equity : ", "load wb KO : " + fund + str(e))

                ws = wb["DATA"]
                max_row = ws.max_row #CLEAN THE SHEET
                max_column = ws.max_column
                for r_idx in range(1,max_row+100):
                    for c_idx in range(1,max_column + 45):
                        ws.cell(row=r_idx, column=c_idx, value='')


                rows = dataframe_to_rows(screen_final, index=False)
                for r_idx, row in enumerate(rows, 1):
                    for c_idx, value in enumerate(row, 1):
                        ws.cell(row=r_idx, column=c_idx, value=value)


                #### FEUILLE FOND IA ######
                # Formater l'année et le mois au format AAAAMM


                df_ml_explicativite = pd.read_excel(path_exp_ml_world)


                ws = wb["DATA_IA"]
                max_row = ws.max_row #CLEAN THE SHEET
                max_column = ws.max_column
                for r_idx in range(1,max_row+100):
                    for c_idx in range(1,max_column + 45):
                        ws.cell(row=r_idx, column=c_idx, value='')


                rows = dataframe_to_rows(df_ml_explicativite, index=False)
                for r_idx, row in enumerate(rows, 1):
                    for c_idx, value in enumerate(row, 1):
                        ws.cell(row=r_idx, column=c_idx, value=value)

                ws = wb["IA"]
                ws.cell(row=2, column=2, value=fund_full.iloc[0,0])
                
                ########

                ws = wb["Fonds"]
                max_row = ws.max_row #CLEAN THE SHEET
                for r_idx in range(1,max_row+100):
                    for c_idx in range(1,45):
                        ws.cell(row=r_idx, column=c_idx, value='')
                



                cols=["Dividend Avg Percentile","Value Avg Percentile","Quality Avg Percentile","Mom Avg Percentile","LowVol Avg Percentile","Growth Avg Percentile","Beta"]
                fund_full[cols] = fund_full[cols].round(2)
                # fund_full = fund_full.to_string(formatters={
                #     'Contrib TE': '{:,.2f}'.format,
                # })
                rows = dataframe_to_rows(fund_full, index=False)
                for r_idx, row in enumerate(rows, 1):
                    for c_idx, value in enumerate(row, 1):
                        ws.cell(row=r_idx, column=c_idx, value=value)
                        if c_idx==16:
                            ws.cell(row=r_idx, column=c_idx).number_format = '0.00%'   #TE
                        if r_idx>1:
                            ws.cell(row=r_idx, column=c_idx).alignment = Alignment(horizontal='center')

                ws = wb["Analyse"]    
                ws.cell(row=1, column=4, value=fund)      
                ws.cell(row=1, column=5, value=mapping_fund_ind2[fund])
                ws.cell(row=2, column=17, value=te_ptf)


                    
                ws = wb["Analyse"]      
                hors_indice_df=hors_indice_df.sort_values(by=['%ACTIF'], ascending=False)
                rows = dataframe_to_rows(hors_indice_df, index=False, header= False)
                for r_idx, row in enumerate(rows, 1):
                    for c_idx, value in enumerate(row, 1):
                        ws.cell(row=57+r_idx, column=1+c_idx, value=value)
                ws.cell(row=56,column=4, value=hors_indice_df['%ACTIF'].sum())       


                worst_ml_stocks_df = fund_full[fund_full["Score ML"]<1][["LIBELLE", "Score ML"]].sort_values(by=["Score ML"], ascending = True)
                rows = dataframe_to_rows(worst_ml_stocks_df, index=False, header= False)
                for r_idx, row in enumerate(rows, 1):
                    for c_idx, value in enumerate(row, 1):
                        ws.cell(row=57+r_idx, column=7+c_idx, value=value)


                df_devation_stocks = pd.merge(dict_bench2[mapping_fund_ind2[fund]].rename(columns = {"%ACTIF" : "Weight Bench"})[["ISIN", "Weight Bench", "LIBELLE"]], fund_full[["ISIN", "%ACTIF 100%"]], on = "ISIN", how = "left").fillna(0)
                df_devation_stocks["Weight Bench"]/= df_devation_stocks["Weight Bench"].sum()
                df_devation_stocks["Deviation"] = df_devation_stocks["%ACTIF 100%"] - df_devation_stocks["Weight Bench"]
                df_devation_stocks = df_devation_stocks.sort_values("Deviation")
                surexpo = df_devation_stocks.iloc[-5:]["Deviation"].sum()
                sousexpo = df_devation_stocks.iloc[:5]["Deviation"].sum()
                top_stock_surexpo = df_devation_stocks.iloc[-5:][["LIBELLE", "Deviation"]]
                top_stock_sousexpo = df_devation_stocks.iloc[:5][["LIBELLE", "Deviation"]]

                rows = dataframe_to_rows(top_stock_surexpo, index=False, header= False)
                for r_idx, row in enumerate(rows, 1):
                    for c_idx, value in enumerate(row, 1):
                        ws.cell(row=57+r_idx, column=11+c_idx, value=value)

                ws.cell(row=56, column=13, value=surexpo)

                rows = dataframe_to_rows(top_stock_sousexpo, index=False, header= False)
                for r_idx, row in enumerate(rows, 1):
                    for c_idx, value in enumerate(row, 1):
                        ws.cell(row=57+r_idx, column=15+c_idx, value=value)

                ws.cell(row=56, column=17, value=sousexpo)

                ws = wb["Benchmark"]
                max_row = ws.max_row #CLEAN THE SHEET
                for r_idx in range(1,max_row+2000):
                    for c_idx in range(1,45):
                        ws.cell(row=r_idx, column=c_idx, value='')
                            
                rows = dataframe_to_rows(indice_full, index=False)
                for r_idx, row in enumerate(rows, 1):
                    for c_idx, value in enumerate(row, 1):
                        ws.cell(row=r_idx, column=c_idx, value=value)


                ###############################################################################################
                ###############################################################################################
                #'NEWS'
                try:
                    ISIN_f=list(fund_full.ISIN.unique())
                    NEWS=pd.read_parquet(Path_news)

                    today = datetime.today().date()
                    five_days_ago = pd.Timestamp(today - timedelta(days=7),tz='UTC')

                    NEWS_last7D = NEWS[NEWS["story_time"] >= five_days_ago]
                    filtered_NEWS = NEWS_last7D[NEWS_last7D['ISIN'].isin(ISIN_f)]
                    filtered_NEWS=filtered_NEWS.merge(fund_full[["ISIN","LIBELLE"]],on="ISIN",how='left')
                    filtered_NEWS=filtered_NEWS[["ISIN","LIBELLE","story_time","headlines","subjects","story_body_clean"]].sort_values(by="story_time",ascending=False)

                    try:
                        filtered_NEWS['subjects'] = filtered_NEWS['subjects'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
                    except Exception as e:
                        error=''

                    filtered_NEWS['story_time'] = pd.to_datetime(filtered_NEWS['story_time'], errors='coerce')
                    filtered_NEWS['story_time'] = filtered_NEWS['story_time'].dt.tz_localize(None)

                    ws = wb["NewsFlow"]
                    for r_idx in range(1,2000): #CLEAN THE SHEET
                        for c_idx in range(1,45):
                            ws.cell(row=r_idx, column=c_idx, value='')
                                
                    rows = dataframe_to_rows(filtered_NEWS, index=False)
                    for r_idx, row in enumerate(rows, 1):
                        for c_idx, value in enumerate(row, 1):
                            ws.cell(row=r_idx, column=c_idx, value=value)

                except Exception as e:
                    sendNotification("jobprod",  "EQUITY REPORT ERROR ON NEWS on " +str(fund), str(e))
                    print("EQUITY REPORT ERROR ON NEWS on " + str(fund) + "  " + str(e))



                ###############################################################################################
                ###############################################################################################
                #'TOP WORST'
                # Define the start dates for each period
                df_returns.index = pd.to_datetime(df_returns.index)
                current_date = df_returns.index.max()
                dates_last_year = df_returns.index.year == datetime.now().year-1

                dates_last_year=max(df_returns.index[dates_last_year])# For YTD
                one_week_start = current_date - pd.Timedelta(days=7)
                one_month_start = current_date - pd.Timedelta(days=30)
                one_year_start = current_date - pd.Timedelta(days=365)



                base = fund_full[["ISIN","%ACTIF"]]
                base2 = indice_full[["ISIN","%ACTIF 100%"]]
                base = base.rename(columns = {"%ACTIF" : "Weight"})
                base2 = base2.rename(columns = {"%ACTIF 100%": "Weight"})
                base = base.merge(last_screen[["ISIN","Company SEDOL","Name"]],on=["ISIN"],how='left')
                base2 = base2.merge(last_screen[["ISIN","Company SEDOL","Name"]],on=["ISIN"],how='left')
                base = base.merge(last_screen[["ISIN", sector_col]],on=["ISIN"],how='left')
                base2 = base2.merge(last_screen[["ISIN", sector_col]],on=["ISIN"],how='left')

                # Calculate performance for all ISINs for each period
                perf_1w = calculate_cumulative_return(df_returns, one_week_start, current_date)
                perf_1m = calculate_cumulative_return(df_returns, one_month_start, current_date)
                perf_ytd = calculate_cumulative_return(df_returns, dates_last_year, current_date)
                perf_1y = calculate_cumulative_return(df_returns, one_year_start, current_date)

                base['perf_1w'] = base["Company SEDOL"].map(perf_1w)
                base['perf_1m'] = base["Company SEDOL"].map(perf_1m)
                base['perf_ytd'] = base["Company SEDOL"].map(perf_ytd)
                base['perf_1y'] = base["Company SEDOL"].map(perf_1y)

                base['perf_1w'] = base['perf_1w'].round(4)
                base['perf_1m'] = base['perf_1m'].round(4)
                base['perf_ytd'] = base['perf_ytd'].round(4)
                base['perf_1y'] = base['perf_1y'].round(4)
                del base["Company SEDOL"]
                base = base.sort_values(by=['perf_1w'], ascending=False)

                exclude_list = ['NOK','USD','DKK','GBP','EUR','CHF','SEK','JPY','CNY','KRY','SGD','CAD','AUD','ZAR']
                base = base[~base['ISIN'].isin(exclude_list)]


                base2['perf_1w'] = base2["Company SEDOL"].map(perf_1w)
                base2['perf_1m'] = base2["Company SEDOL"].map(perf_1m)
                base2['perf_ytd'] = base2["Company SEDOL"].map(perf_ytd)
                base2['perf_1y'] = base2["Company SEDOL"].map(perf_1y)

                base2['perf_1w'] = base2['perf_1w'].round(4)
                base2['perf_1m'] = base2['perf_1m'].round(4)
                base2['perf_ytd'] = base2['perf_ytd'].round(4)
                base2['perf_1y'] = base2['perf_1y'].round(4)
                del base2["Company SEDOL"]
                base2 = base2.sort_values(by=['perf_1w'], ascending=False)
                base2.dropna(subset=['perf_1w'],inplace=True)







                Top=base.head(10)
                Top2=base2.head(20)
                # Top = Top.to_string(formatters={
                #     'perf_1w': '{:,.2f}'.format,
                #     'perf_1m': '{:,.2f}'.format,
                #     'perf_ytd': '{:,.2%}'.format,
                #     'perf_1y': '{:,.2%}'.format
                # })

                Worst=base.tail(10)
                Worst2=base2.tail(20)
                # Worst = Worst.to_string(formatters={
                #     'perf_1w': '{:,.2f}'.format,
                #     'perf_1m': '{:,.2f}'.format,
                #     'perf_ytd': '{:,.2%}'.format,
                #     'perf_1y': '{:,.2%}'.format
                # })

                print(Top)
                print(Worst)



                ws = wb["TopWorst Perf"]
                # max_row = 100 #CLEAN THE SHEET
                # for r_idx in range(1,100):
                #     for c_idx in range(1,50):
                #         ws.cell(row=r_idx, column=c_idx, value='')

                # PTF
                rows = dataframe_to_rows(Top, index=False)
                for r_idx, row in enumerate(rows, 1):
                    for c_idx, value in enumerate(row, 1):
                        ws.cell(row=r_idx+3, column=c_idx+2, value=value)
                        if c_idx>4 or c_idx==2:
                            ws.cell(row=r_idx+3, column=c_idx+2).number_format = '0.00%'   #TE
                        if r_idx>1:
                            ws.cell(row=r_idx+3, column=c_idx+2).alignment = Alignment(horizontal='center')

                rows = dataframe_to_rows(Worst, index=False)
                for r_idx, row in enumerate(rows, 1):
                    for c_idx, value in enumerate(row, 1):
                        ws.cell(row=r_idx+3, column=c_idx+11, value=value)
                        if c_idx>4 or c_idx==2:
                            ws.cell(row=r_idx+3, column=c_idx+11).number_format = '0.00%'   #TE
                        if r_idx>1:
                            ws.cell(row=r_idx+3, column=c_idx+11).alignment = Alignment(horizontal='center')


                # BENCH
                rows = dataframe_to_rows(Top2, index=False)
                for r_idx, row in enumerate(rows, 1):
                    for c_idx, value in enumerate(row, 1):
                        ws.cell(row=r_idx+19, column=c_idx+2, value=value)
                        if c_idx>4 or c_idx==2:
                            ws.cell(row=r_idx+19, column=c_idx+2).number_format = '0.000%'   #TE
                        if r_idx>1:
                            ws.cell(row=r_idx+19, column=c_idx+2).alignment = Alignment(horizontal='center')

                rows = dataframe_to_rows(Worst2, index=False)
                for r_idx, row in enumerate(rows, 1):
                    for c_idx, value in enumerate(row, 1):
                        ws.cell(row=r_idx+19, column=c_idx+11, value=value)
                        if c_idx>4 or c_idx==2:
                            ws.cell(row=r_idx+19, column=c_idx+11).number_format = '0.000%'   #TE
                        if r_idx>1:
                            ws.cell(row=r_idx+19, column=c_idx+11).alignment = Alignment(horizontal='center')



                # PRINT DS DOSSIER INGEFI ET GESTION ACTION
                try:
                    Output_File=os.path.join(wb_output_folder_path, str(fund) + "_" + today_str + ".xlsx")
                    wb.save(Output_File)
                except Exception as e:
                    sendNotification("jobprod",  "EQUITY REPORT Generated KO ds INGEFI : " + str(fund), str(e))
                    print("EQUITY REPORT ERROR ON NEWS on " + str(fund) + "  " + str(e))

                try:
                    if equipe =="Epargne_Salariale":
                        Output_File2=os.path.join(output_path_equipe_ES, str(fund) + "_" + today_str + ".xlsx")
                    else:
                        Output_File2=os.path.join(output_path_equipe_caroline, str(fund) + "_" + today_str + ".xlsx")
                    wb.save(Output_File2)
                except Exception as e:
                    sendNotification("jobprod",  "EQUITY REPORT Generated KO ds Gestion Action : " + str(fund), str(e))

                msg2 = msg2 + "Generated OK : " + str(fund) + "\n\n"
                print("Generated OK : " + fund)
        
        except Exception as e:
            sendNotification("jobprod",  "Generate Report Equity : ", "Generated KO : " + fund + str(e))
            print("Generated KO : " + fund,e)

    if msg2 !="":
        sendNotification("jobprod",  "Generate Report Equity OK ds IngeFi et Gestion Action", msg2)


if __name__ == '__main__':
    generate_report(sector_col=" Benchmark ICB Supersector ")
