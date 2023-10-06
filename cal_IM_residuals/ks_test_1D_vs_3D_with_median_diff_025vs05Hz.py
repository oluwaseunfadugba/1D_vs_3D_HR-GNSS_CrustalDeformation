#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:33:25 2022

@author: oluwaseunfadugba
"""
# Helper functions
#%% extract_ks_result   
def extract_ks_result(f_data1,f_data2,attribute):
    import numpy as np
    import scipy.stats as stats
    
    ks_hypo = []
    ks_stat = []
    ks_pval = []
    ks_Dcrit = []
    ks_nx = []
    ks_ny = []
    ks_median_x = []
    ks_median_y = []
    
    uniq_hypo = np.intersect1d(np.unique(f_data1['hypdist_index']), np.unique(f_data2['hypdist_index']))
    
    ks_all = stats.kstest(f_data1[attribute], f_data2[attribute])
    nx = len(f_data1[attribute])
    ny = len(f_data2[attribute])
    Dcrit_all = 1.36*np.sqrt((1/nx)+(1/ny))
    
    median_all_1 = f_data1[attribute].median()
    median_all_2 = f_data2[attribute].median()
    
    #print(median_all_1,median_all_2)
    
    ks_hypo.append(0)
    ks_stat.append(ks_all[0])
    ks_pval.append(ks_all[1])
    ks_Dcrit.append(Dcrit_all)
    ks_nx.append(nx)
    ks_ny.append(ny)
    ks_median_x.append(median_all_1)
    ks_median_y.append(median_all_2)
    
    for i in range(len(uniq_hypo)):
        f_data1_hypo = f_data1[f_data1['hypdist_index']==uniq_hypo[i]]
        f_data2_hypo = f_data2[f_data2['hypdist_index']==uniq_hypo[i]]
        
        ks_results_hypo = stats.kstest(f_data1_hypo[attribute].to_numpy(), f_data2_hypo[attribute].to_numpy())
        
        nx = len(f_data1_hypo[attribute])
        ny = len(f_data2_hypo[attribute])
        Dcrit = 1.36*np.sqrt((1/nx)+(1/ny))
        
        medx = f_data1_hypo[attribute].median()
        medy = f_data2_hypo[attribute].median()
        
        ks_hypo.append(uniq_hypo[i])
        ks_stat.append(ks_results_hypo[0])
        ks_pval.append(ks_results_hypo[1])
        ks_Dcrit.append(Dcrit)
        ks_nx.append(nx)
        ks_ny.append(ny)
        ks_median_x.append(medx)
        ks_median_y.append(medy)
    
    return ks_hypo,ks_stat,ks_pval,ks_Dcrit,ks_nx,ks_ny,ks_median_x,ks_median_y

#%% gen_flatfile_kstest
def gen_flatfile_kstest(models,attribute):
    
    import pandas as pd
    import numpy as np
    
    print('Calculating ks-test for '+ attribute)
    print(' ')
    pgd_ks_hypo = []
    pgd_ks_stat = []
    pgd_ks_pval = []
    pgd_ks_Dcrit = []
    pgd_ks_nx = []
    pgd_ks_ny = []
    pgd_tag = []
    pgd_snr = []
    pgd_median_x = []
    pgd_median_y = []
    
    for i in range(len(models[:,0])):
    
        flatfile_data1 = models[i,0]
        flatfile_data2 = models[i,1]
        tag = list(models[i,2].split(" "))
        SNR_threshold = models[i,3]
        n_rupt = models[i,4]
        
        f_data1 = pd.read_csv(flatfile_data1) 
        f_data2 = pd.read_csv(flatfile_data2) 
        
        f_data1 = f_data1[f_data1['rupt_no'].isin(n_rupt)]
        f_data2 = f_data2[f_data2['rupt_no'].isin(n_rupt)]
                  
        for j in range(len(SNR_threshold)):
            f_data1 = f_data1[f_data1['SNR_obs']>=SNR_threshold[j]]
            f_data2 = f_data2[f_data2['SNR_obs']>=SNR_threshold[j]]
            
            if len(f_data1)==0 or len(f_data2)==0:
                return
        
            ks_hypo,ks_stat,ks_pval,ks_Dcrit,ks_nx,ks_ny,ks_median_x,ks_median_y = extract_ks_result(f_data1,f_data2,attribute)
 
            snr = list(str(SNR_threshold[j]).split(" "))
            
            pgd_ks_hypo = pgd_ks_hypo + ks_hypo
            pgd_ks_stat = pgd_ks_stat + ks_stat
            pgd_ks_pval = pgd_ks_pval + ks_pval
            pgd_ks_Dcrit = pgd_ks_Dcrit + ks_Dcrit
            pgd_ks_nx = pgd_ks_nx + ks_nx
            pgd_ks_ny = pgd_ks_ny + ks_ny
            pgd_median_x = pgd_median_x + ks_median_x
            pgd_median_y = pgd_median_y + ks_median_y
            pgd_tag = pgd_tag + [tag[i//len(ks_hypo)] for i in range(len(ks_hypo))]
            pgd_snr = pgd_snr + [snr[i//len(ks_hypo)] for i in range(len(ks_hypo))]

    if len(f_data1)==0 or len(f_data2)==0:
        return
    else:
        flatfile_df = pd.DataFrame({"tag":pgd_tag, 
                       "ks_hypo":pgd_ks_hypo,
                       "ks_stat":pgd_ks_stat,
                       "ks_pval":pgd_ks_pval,
                       "ks_Dcrit":pgd_ks_Dcrit,
                       "ks_nx":pgd_ks_nx,
                       "ks_ny":pgd_ks_ny,
                       "ks_median_x":pgd_median_x,
                       "ks_median_y":pgd_median_y,
                       #"ks_n_min":min(pgd_ks_nx,pgd_ks_ny),
                       "ks_n":np.add(pgd_ks_nx,pgd_ks_ny),
                       "SNR_thresh":pgd_snr})
    
        flatfile_df['tag'] = flatfile_df['tag'].astype(str)#.zfill(5)
    
        flatfile_df.to_csv("flatfile_ks_test_"+attribute+".csv",index=False)
    
    return
#%% plot_ks_test_driv
def old_plot_ks_test_driv(models,attribute):
    
    
    plot_ks_test_all_w_median("flatfile_ks_test_"+attribute+".csv",attribute)
    
    #plot_ks_test_all("flatfile_ks_test_"+attribute+".csv",attribute)
    # for i in range(len(models[:,0])):
    #     #plot_ks_test_each("flatfile_ks_test_"+attribute+".csv", attribute,str(models[i,2]))
    #     plot_ks_test_each_with_median("flatfile_ks_test_"+attribute+".csv", attribute,str(models[i,2]))
    
    return
#%% plot_ks_test_all
def plot_ks_test_all(flatfile,attribute):
    import pandas as pd
    
    df = pd.read_csv(flatfile) 
    df = df[(df['ks_hypo']==0) & (df['SNR_thresh']==0)].reset_index(drop=True)
    axes_max = np.ceil(max(np.abs( np.concatenate((df['ks_stat'].to_numpy(),df['ks_Dcrit'].to_numpy(),df['ks_pval'].to_numpy())) ))*2)/2
  
    factor = 1.5
    bf = 0.25
    fig, axes = plt.subplots(figsize=(30, 20))
    fig.suptitle('K-S Test Results between 1D vs 3D ('+attribute+')', fontsize=tick_font+20)

    axes.plot(df['tag'],df['ks_stat'],'-bo',lw=7,label='ks-stat')
    plt.scatter(df['tag'],df['ks_stat'],c='b',s = df['ks_n']*1.5)
    axes.plot(df['tag'],df['ks_Dcrit'],'--bo',lw=7,label='D Critical') #,ms = 20
    
    axes.plot(df['tag'],df['ks_pval'],'-ro',lw=7,label='p value')
    plt.scatter(df['tag'],df['ks_pval'],c='r',s = df['ks_n']*factor)
    axes.axhline(y=0.05, c= 'r',ls='--',linewidth=5,label='pval = 0.05')
    
    plt.scatter(df['tag'][0], axes_max+bf-0.25+0.04, s=round(max(df['ks_n']*factor)), c='b')
    plt.scatter(df['tag'][0], axes_max+bf-0.45+0.04, s=round((df['ks_n'].mean()*factor)), c='b')
    plt.text(0.15, axes_max+bf-0.25,"nx + ny = "+str(round(max(df['ks_n']*factor))), fontsize=tick_font)
    plt.text(0.15, axes_max+bf-0.45,"nx + ny = "+str(round((df['ks_n'].mean()*factor))), fontsize=tick_font)
    
    axes.tick_params(axis='x',labelsize=tick_font+10)#,labelrotation=60,length=20, width=3)
    plt.setp(axes.get_xticklabels(),rotation=60, ha="right",rotation_mode="anchor")
    axes.tick_params(axis='y',labelsize=tick_font+10,labelrotation=0,length=20, width=3)
    axes.set_xlabel('Simulations', fontsize=tick_font+10)
    axes.set_ylabel('value', fontsize=tick_font+10)
    axes.set(ylim=(-axes_max-bf, axes_max+bf))
    axes.legend(loc='best',fontsize=tick_font)
    plt.grid()

    figpath = os.getcwd() +'/fig.kstest_all_'+attribute+'.IM_residuals.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)

    plt.close()  

    return

#%% plot_ks_test_all_w_median
def old_plot_ks_test_all_w_median(flatfile,attribute):
    import pandas as pd
    import numpy as np

    df = pd.read_csv(flatfile) 
    df = df[(df['ks_hypo']==0) & (df['SNR_thresh']==0)].reset_index(drop=True)
    axes_max = np.ceil(max(np.abs( np.concatenate((df['ks_stat'].to_numpy(),df['ks_Dcrit'].to_numpy(),df['ks_pval'].to_numpy())) ))*2)/2
    
    factor = 1.5
    bf = 0.25
    fig, axes = plt.subplots(figsize=(30, 20))
    fig.suptitle('K-S Test Results between 1D vs 3D ('+attribute+')', fontsize=tick_font+20)
    
    axes.plot(df['tag'],df['ks_stat'],'-bo',lw=7,label='ks-stat')
    plt.scatter(df['tag'],df['ks_stat'],c='b',s = df['ks_n']*1.5)
    axes.plot(df['tag'],df['ks_Dcrit'],'--bo',lw=7,label='D Critical') #,ms = 20
    
    axes.plot(df['tag'],df['ks_pval'],'-ro',lw=7,label='p value')
    plt.scatter(df['tag'],df['ks_pval'],c='r',s = df['ks_n']*factor)
    axes.axhline(y=0.05, c= 'r',ls='--',linewidth=5,label='pval = 0.05')
    
    plt.scatter(df['tag'][0], axes_max+bf-0.25+0.04, s=round(max(df['ks_n']*factor)), c='b')
    plt.scatter(df['tag'][0], axes_max+bf-0.45+0.04, s=round((df['ks_n'].mean()*factor)), c='b')
    plt.text(0.15, axes_max+bf-0.25,"nx + ny = "+str(round(max(df['ks_n']*factor))), fontsize=tick_font)
    plt.text(0.15, axes_max+bf-0.45,"nx + ny = "+str(round((df['ks_n'].mean()*factor))), fontsize=tick_font)
    
    axes.tick_params(axis='x',labelsize=tick_font+10)#,labelrotation=60,length=20, width=3)
    plt.setp(axes.get_xticklabels(),rotation=60, ha="right",rotation_mode="anchor")
    axes.tick_params(axis='y',labelsize=tick_font+10,labelrotation=0,length=20, width=3)
    axes.set_xlabel('Simulations', fontsize=tick_font+10)
    axes.set_ylabel('value', fontsize=tick_font+10)
    axes.set(ylim=(-axes_max-bf, axes_max+bf))
    axes.legend(bbox_to_anchor=(1.2, 1), loc='upper left', borderaxespad=0,fontsize=tick_font)
    
    axes2=axes.twinx()
    
    axes2.plot(df['tag'],np.abs(df['ks_median_y'])-np.abs(df['ks_median_x']),'-go',lw=7,label='med_3D-1D')
    axes2.scatter(df['tag'],np.abs(df['ks_median_y'])-np.abs(df['ks_median_x']),c='g',s = df['ks_n']*1.5)
    axes2.set_ylabel('3D-1D Res Median', fontsize=tick_font+10)
    axes2.axhline(y=0, c= 'g',ls='--',linewidth=5,label='3D-1D Res = 0')
    axes2.tick_params(axis='y',labelsize=tick_font+10,labelrotation=0,length=20, width=3)
    axes2_max = np.ceil(max(np.abs(np.abs(df['ks_median_y'])-np.abs(df['ks_median_x']))*2))/2
    
    axes2.set(ylim=(-axes2_max-bf, axes2_max+bf))
    axes2.legend(bbox_to_anchor=(1.2, 0.6), loc='upper left', borderaxespad=0,fontsize=tick_font)
    plt.grid()

    figpath = os.getcwd() +'/fig.kstest_all_median_'+attribute+'.IM_residuals.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)

    plt.close()  

    return

#%% plot_ks_test_each
def plot_ks_test_each(flatfile, attribute,tag):
    import pandas as pd

    df = pd.read_csv(flatfile) 
    df = df[(df['tag']==tag) & (df['ks_hypo']!=0) & (df['SNR_thresh']==0)].reset_index(drop=True)
    axes_max = np.ceil(max(np.abs( np.concatenate((df['ks_stat'].to_numpy(),df['ks_Dcrit'].to_numpy(),df['ks_pval'].to_numpy())) ))*2)/2
    
    factor = 4
    bf = 0.25
    fig, axes = plt.subplots(figsize=(30, 20))
    fig.suptitle('K-S Test Results for '+tag +' ('+attribute+')', fontsize=tick_font+20)
    
    axes.plot(df['ks_hypo'],df['ks_stat'],'-bo',lw=7,label='ks-stat wrt dist')
    plt.scatter(df['ks_hypo'],df['ks_stat'],c='b',s = df['ks_n']*factor)
    axes.plot(df['ks_hypo'],df['ks_Dcrit'],'--bo',lw=7,label='D Critical wrt dist') #,ms = 20
    
    axes.plot(df['ks_hypo'],df['ks_pval'],'-ro',lw=7,label='p value wrt dist')
    plt.scatter(df['ks_hypo'],df['ks_pval'],c='r',s = df['ks_n']*factor)
    axes.axhline(y=0.05, c= 'r',ls='--',linewidth=5,label='pval = 0.05')
    
    plt.scatter(df['ks_hypo'][0], axes_max+bf-0.25+0.04, s=round(max(df['ks_n']*factor)), c='b')
    plt.scatter(df['ks_hypo'][0], axes_max+bf-0.45+0.04, s=round((df['ks_n'].mean()*factor)), c='b')
    plt.text((df['ks_hypo'][0]+df['ks_hypo'][1])/2, axes_max+bf-0.25,"nx + ny = "+\
             str(round(max(df['ks_n']*1.5))), fontsize=tick_font)
    plt.text((df['ks_hypo'][0]+df['ks_hypo'][1])/2, axes_max+bf-0.45,"nx + ny = "+\
             str(round((df['ks_n'].mean()*1.5))), fontsize=tick_font)
    
    axes.tick_params(axis='x',labelsize=tick_font+10,labelrotation=45,length=20, width=3)
    axes.tick_params(axis='y',labelsize=tick_font+10,labelrotation=0,length=20, width=3)
    axes.set_xlabel('distance (km)', fontsize=tick_font+10)
    axes.set_ylabel('value', fontsize=tick_font+10)
    axes.set(ylim=(-axes_max-bf, axes_max+bf))
    axes.legend(bbox_to_anchor=(1.2, 1), loc='upper left', borderaxespad=0,fontsize=tick_font)
    plt.grid()
      
    figpath = os.getcwd() +'/fig.kstest_'+tag+'_'+attribute+'.IM_residuals.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    
    plt.close()  
    
    return

#%% plot_ks_test_each_with_median
def plot_ks_test_each_with_median(flatfile, attribute,tag):
    import pandas as pd

    df = pd.read_csv(flatfile) 
    df = df[(df['tag']==tag) & (df['ks_hypo']!=0) & (df['SNR_thresh']==0)].reset_index(drop=True)
    axes_max = np.ceil(max(np.abs( np.concatenate((df['ks_stat'].to_numpy(),df['ks_Dcrit'].to_numpy(),df['ks_pval'].to_numpy())) ))*2)/2
    
    factor = 4
    bf = 0.25
    fig, axes = plt.subplots(figsize=(30, 20))
    fig.suptitle('K-S Test Results for '+tag +' ('+attribute+')', fontsize=tick_font+20)
    
    axes.plot(df['ks_hypo'],df['ks_stat'],'-bo',lw=7,label='ks-stat wrt dist')
    plt.scatter(df['ks_hypo'],df['ks_stat'],c='b',s = df['ks_n']*factor)
    axes.plot(df['ks_hypo'],df['ks_Dcrit'],'--bo',lw=7,label='D Critical wrt dist') #,ms = 20
    
    axes.plot(df['ks_hypo'],df['ks_pval'],'-ro',lw=7,label='p value wrt dist')
    plt.scatter(df['ks_hypo'],df['ks_pval'],c='r',s = df['ks_n']*factor)
    axes.axhline(y=0.05, c= 'r',ls='--',linewidth=5,label='pval = 0.05')
    
    plt.scatter(df['ks_hypo'][0], axes_max+bf-0.25+0.04, s=round(max(df['ks_n']*factor)), c='b')
    plt.scatter(df['ks_hypo'][0], axes_max+bf-0.45+0.04, s=round((df['ks_n'].mean()*factor)), c='b')
    plt.text((df['ks_hypo'][0]+df['ks_hypo'][1])/2, axes_max+bf-0.25,"nx + ny = "+\
             str(round(max(df['ks_n']*1.5))), fontsize=tick_font)
    plt.text((df['ks_hypo'][0]+df['ks_hypo'][1])/2, axes_max+bf-0.45,"nx + ny = "+\
             str(round((df['ks_n'].mean()*1.5))), fontsize=tick_font)
    
    axes.tick_params(axis='x',labelsize=tick_font+10,labelrotation=45,length=20, width=3)
    axes.tick_params(axis='y',labelsize=tick_font+10,labelrotation=0,length=20, width=3)
    axes.set_xlabel('distance (km)', fontsize=tick_font+10)
    axes.set_ylabel('value', fontsize=tick_font+10)
    
    axes.set(ylim=(-axes_max-bf, axes_max+bf))
    axes.legend(bbox_to_anchor=(1.2, 1), loc='upper left', borderaxespad=0,fontsize=tick_font)
       
    axes2=axes.twinx()
    
    axes2.plot(df['ks_hypo'],np.abs(df['ks_median_y'])-np.abs(df['ks_median_x']),'-go',lw=7,label='med_3D-1D')
    axes2.scatter(df['ks_hypo'],np.abs(df['ks_median_y'])-np.abs(df['ks_median_x']),c='g',s = df['ks_n']*1.5)
    axes2.set_ylabel('3D-1D Res Median', fontsize=tick_font+10)
    axes2.axhline(y=0, c= 'g',ls='--',linewidth=5,label='3D-1D Res = 0')
    axes2.tick_params(axis='y',labelsize=tick_font+10,labelrotation=0,length=20, width=3)
    axes2_max = np.ceil(max(np.abs(np.abs(df['ks_median_y'])-np.abs(df['ks_median_x']))*2))/2
    
    axes2.set(ylim=(-axes2_max-bf, axes2_max+bf))
    axes2.legend(bbox_to_anchor=(1.2, 0.6), loc='upper left', borderaxespad=0,fontsize=tick_font)
    
    plt.grid()
    
    figpath = os.getcwd() +'/fig.kstest_median_'+tag+'_'+attribute+'.IM_residuals.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    
    plt.close()  
    
    return

#%% plot_pval_all
def plot_pval_all(models,attribute,pltlegend=0):
    import pandas as pd
    
    flatfile = "flatfile_ks_test_"+attribute+".csv"
  
    factor = 4
    bf = 0.25
    lw = 12
    fig, axes = plt.subplots(figsize=(30, 20))
    fig.suptitle('P-values ('+attribute+')', fontsize=tick_font+20)
    
    for i in range(len(models[:,0])):
        
        df = pd.read_csv(flatfile) 
        df = df[(df['tag']==str(models[i,2])) & (df['ks_hypo']!=0) & (df['SNR_thresh']==0)].reset_index(drop=True)
        
        #print(df)
        axes.plot(df['ks_hypo'],df['ks_pval'],lw=lw,ls = ls[i],label=str(models[i,2]))
        plt.scatter(df['ks_hypo'],df['ks_pval'],s = df['ks_n']*factor)
    
    axes.axhline(y=0.05, c= 'k',ls='--',linewidth=lw+2,label='pval = 0.05')    
    axes.tick_params(axis='x',labelsize=tick_font+10,labelrotation=0,length=20, width=3)
    axes.tick_params(axis='y',labelsize=tick_font+10,labelrotation=0,length=20, width=3)
    axes.set_xlabel('distance (km)', fontsize=tick_font+10)
    axes.set_ylabel('p-value', fontsize=tick_font+10)
    axes.set(ylim=(-0.1, 1.1))
    axes.set(xlim=(0, 1600))
    if pltlegend == 1:
        axes.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0,fontsize=tick_font)
        figpath = os.getcwd() +'/fig.kstest_pval_w_dist_'+attribute+'.legend.IM_residuals.png'
    else:
        figpath = os.getcwd() +'/fig.kstest_pval_w_dist_'+attribute+'.IM_residuals.png'
        
    #plt.grid()
    axes.grid(color = 'k', linestyle = '-', linewidth = 2)
    [i.set_linewidth(4) for i in axes.spines.values()]
    
    plt.savefig(figpath, bbox_inches='tight', dpi=100)
    
    plt.close()  

#%% old_plot_pval_all
def old_plot_pval_all(models,attribute):
    import pandas as pd
    
    flatfile = "flatfile_ks_test_"+attribute+".csv"
  
    factor = 4
    bf = 0.25
    fig, axes = plt.subplots(figsize=(30, 20))
    fig.suptitle('P-values with distance for all simulations ('+attribute+')', fontsize=tick_font+20)
    
    for i in range(len(models[:,0])):
        
        df = pd.read_csv(flatfile) 
        df = df[(df['tag']==str(models[i,2])) & (df['ks_hypo']!=0) & (df['SNR_thresh']==0)].reset_index(drop=True)

        axes.plot(df['ks_hypo'],df['ks_pval'],lw=7,label=str(models[i,2]))
        plt.scatter(df['ks_hypo'],df['ks_pval'],s = df['ks_n']*factor)
          
        # plt.scatter(df['ks_hypo'][0], 1.4, s=round(max(df['ks_n']*factor)), c='b')
        # plt.scatter(df['ks_hypo'][0], 1.2, s=round((df['ks_n'].mean()*factor)), c='b')
        # plt.text((df['ks_hypo'][0]+df['ks_hypo'][1])/2, 1.37,"nx + ny = "+\
        #          str(round(max(df['ks_n']*1.5))), fontsize=tick_font)
        # plt.text((df['ks_hypo'][0]+df['ks_hypo'][1])/2, 1.17,"nx + ny = "+\
        #          str(round((df['ks_n'].mean()*1.5))), fontsize=tick_font)
    
    axes.axhline(y=0.05, c= 'r',ls='--',linewidth=5,label='pval = 0.05')    
    axes.tick_params(axis='x',labelsize=tick_font+10,labelrotation=45,length=20, width=3)
    axes.tick_params(axis='y',labelsize=tick_font+10,labelrotation=0,length=20, width=3)
    axes.set_xlabel('distance (km)', fontsize=tick_font+10)
    axes.set_ylabel('value', fontsize=tick_font+10)
    axes.set(ylim=(-0.1, 1.5))
    axes.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0,fontsize=tick_font)
    plt.grid()
    
    
    figpath = os.getcwd() +'/fig.kstest_pval_w_dist_'+attribute+'.IM_residuals.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    
    plt.close() 

#%% plot_median_all
def plot_median_all(models,attribute,pltlegend = 0):
    import pandas as pd
    
    flatfile = "flatfile_ks_test_"+attribute+".csv"

    factor = 4
    bf = 0.25
    lw = 12
    fig, axes = plt.subplots(figsize=(30, 20))
    fig.suptitle('3D-1D Residual Median ('+attribute+')', fontsize=tick_font+20)
    
    for i in range(len(models[:,0])):       
        df = pd.read_csv(flatfile) 
        df = df[(df['tag']==str(models[i,2])) & (df['ks_hypo']!=0) & (df['SNR_thresh']==0)].reset_index(drop=True)
    
        axes.plot(df['ks_hypo'],np.abs(df['ks_median_y'])-np.abs(df['ks_median_x']),lw=lw,ls = ls[i],label=str(models[i,2]))
        plt.scatter(df['ks_hypo'],np.abs(df['ks_median_y'])-np.abs(df['ks_median_x']),s = df['ks_n']*factor)

        # plt.scatter(df['ks_hypo'][0], 1.4, s=round(max(df['ks_n']*factor)), c='b')
        # plt.scatter(df['ks_hypo'][0], 1.2, s=round((df['ks_n'].mean()*factor)), c='b')
        # plt.text((df['ks_hypo'][0]+df['ks_hypo'][1])/2, 1.37,"nx + ny = "+\
        #          str(round(max(df['ks_n']*1.5))), fontsize=tick_font)
        # plt.text((df['ks_hypo'][0]+df['ks_hypo'][1])/2, 1.17,"nx + ny = "+\
        #          str(round((df['ks_n'].mean()*1.5))), fontsize=tick_font)
    
    axes.axhline(y=0, c= 'k',ls='--',linewidth=lw+2,label='3D-1D Res = 0')    
    axes.tick_params(axis='x',labelsize=tick_font+10,labelrotation=0,length=20, width=3)
    axes.tick_params(axis='y',labelsize=tick_font+10,labelrotation=0,length=20, width=3)
    axes.set_xlabel('distance (km)', fontsize=tick_font+10)
    axes.set_ylabel('3D-1D Res Median', fontsize=tick_font+10)
    if pltlegend == 1:
        axes.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0,fontsize=tick_font)
        figpath = os.getcwd() +'/fig.kstest_median_w_dist_'+attribute+'.legend.IM_residuals.png'
    else:
        figpath = os.getcwd() +'/fig.kstest_median_w_dist_'+attribute+'.IM_residuals.png'
    #plt.grid()
    
    #axes.set(xlim=(0, 1100)) #1600
    
    axes.grid(color = 'k', linestyle = '-', linewidth = 2)
    [i.set_linewidth(4) for i in axes.spines.values()]

    plt.savefig(figpath, bbox_inches='tight', dpi=100)
    
    plt.close()  

#%% plot_ks_test_all_w_median
def plot_ks_test_all_w_median(attribute,pltlegend = 0):
    import pandas as pd
    import numpy as np

    flatfile = "flatfile_ks_test_"+attribute+".csv"
    
    df = pd.read_csv(flatfile) 
    df = df[(df['ks_hypo']==0) & (df['SNR_thresh']==0)].reset_index(drop=True)
    # axes_max = np.ceil(max(np.abs( np.concatenate((df['ks_stat'].to_numpy(),
    #                                                df['ks_Dcrit'].to_numpy(),
    #                                                df['ks_pval'].to_numpy())) ))*2)/2
    axes_max = (max(np.abs( np.concatenate((df['ks_stat'].to_numpy(),
                                                   df['ks_Dcrit'].to_numpy(),
                                                   df['ks_pval'].to_numpy()))))*1)/1
    factor = 1.5
    bf = 0#0.15
    lw = 12
    fig, axes = plt.subplots(figsize=(30, 20))
    plt.style.use('tableau-colorblind10') #('seaborn-colorblind') #
    fig.suptitle('K-S Test ('+attribute+')', fontsize=tick_font+20)
    
    axes.plot(df['tag'],df['ks_stat'],c=tcb10[0],ls='-',lw=lw,label='ks-stat') #bo
    plt.scatter(df['tag'],df['ks_stat'],c=tcb10[0],s = df['ks_n']*1.5) #c='b',
    axes.plot(df['tag'],df['ks_Dcrit'],c=tcb10[0],ls='-.',lw=lw,label='D Critical') #,ms = 20 bo
    
    axes.plot(df['tag'],df['ks_pval'],c=tcb10[1],ls='-',lw=lw,label='p value') #ro
    plt.scatter(df['tag'],df['ks_pval'],c=tcb10[1],s = df['ks_n']*factor) #c='r'
    axes.axhline(y=0.05, c='#FF800E',ls='--',linewidth=lw,label='pval = 0.05') #c= 'r',
    

    axes.tick_params(axis='x',labelsize=tick_font-50)#,labelrotation=60,length=20, width=3)
    plt.setp(axes.get_xticklabels(),rotation=30, ha="right",rotation_mode="anchor")
    axes.tick_params(axis='y',labelsize=tick_font+10,labelrotation=0,length=20, width=3)
    axes.set_xlabel('Simulations', fontsize=tick_font+0)
    axes.set_ylabel('k-s test', fontsize=tick_font+0)
    axes.set(ylim=(-axes_max-(axes_max/10), axes_max+(axes_max/10)))
    #axes.set(ylim=(-axes_max, axes_max))
    
    
    [i.set_linewidth(4) for i in axes.spines.values()]
    
    
    #----------------------------------------------
    
    axes2=axes.twinx()
    
    axes2.plot(df['tag'],np.abs(df['ks_median_y'])-np.abs(df['ks_median_x']),c=tcb10[2],ls='-',lw=lw,label='med_3D-1D') #go
    axes2.scatter(df['tag'],np.abs(df['ks_median_y'])-np.abs(df['ks_median_x']),c=tcb10[2],s = df['ks_n']*1.5) #,c='g'
    
    axes2.set_ylabel('3D-1D Res Median', c=tcb10[2],fontsize=tick_font+0)
    #axes2.axhline(y=0, c= 'k',ls='--',linewidth=lw,label='3D-1D Res = 0')
    axes2.tick_params(axis='y',colors=tcb10[2],labelsize=tick_font+10,labelrotation=0,length=20, width=3)
    
    #axes2_max = np.ceil(max(np.abs(np.abs(df['ks_median_y'])-np.abs(df['ks_median_x']))*2))/2
    axes2_max = (max(np.abs(np.abs(df['ks_median_y'])-np.abs(df['ks_median_x'])))*1)/1
    axes2.set(ylim=(-axes2_max-(axes2_max/10), axes2_max+(axes2_max/10)))
    #axes2.set(ylim=(-axes2_max, axes2_max))
    axes2.spines['right'].set_color(tcb10[2])
    

    import matplotlib.ticker
    nticks = 5
    axes.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nticks)) #LinearLocator
    axes2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nticks))

    if pltlegend == 1:
        axes2.scatter(9.1, -axes2_max+(axes2_max/2), s=round(max(df['ks_n']*factor)), c=tcb10[0])
        axes2.scatter(9.1, -axes2_max+(axes2_max/4), s=round(min(df['ks_n']*factor)), c=tcb10[0])
        # plt.text(1.35, 0.4,"nx + ny = "+str(round(max(df['ks_n']*factor))), fontsize=tick_font)
        # axes2.text(9.1, -0.45,"nx + ny = "+str(round((df['ks_n'].mean()*factor))), fontsize=tick_font)
        
        axes.legend(bbox_to_anchor=(1.25, 1), loc='upper left', borderaxespad=0,fontsize=tick_font,frameon=False)
        #axes2.legend(bbox_to_anchor=(1.25, 0.5), loc='upper left', borderaxespad=0,fontsize=tick_font)
        
        legend_elements = [Line2D([0], [0], color=tcb10[2], lw=lw, label='3D-1D Res Median'),
                           Line2D([0], [0], marker='o', color='w', label='n1D+n3D = '+
                                  str(round(max(df['ks_n']))), markerfacecolor='g', 
                                  markersize=round(max(df['ks_n']*factor))/70), #
                           Line2D([0], [0], marker='o', color='w', label='n1D+n3D = '+
                                  str(round(min(df['ks_n']))), markerfacecolor='g', 
                                  markersize=round(min(df['ks_n']*factor))/70)] #
        print(min(df['ks_n']*factor))
        print(max(df['ks_n']*factor))
        print(df['ks_n'])
        axes2.legend(handles=legend_elements, bbox_to_anchor=(1.224, 0.55),
                     loc='upper left',fontsize=tick_font,frameon=False)

        figpath = os.getcwd() +'/fig.kstest_all_median_'+attribute+'.legend.IM_residuals.png'
    
    else:
        axes.grid(color = 'k', linestyle = '-', linewidth = 2)
        axes2.grid(None)
        
        figpath = os.getcwd() +'/fig.kstest_all_median_'+attribute+'.IM_residuals.png'
    
    plt.savefig(figpath, bbox_inches='tight', dpi=100)

    plt.close()  

    return


#%% Driver
import numpy as np
import os
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10') #('seaborn-colorblind') #
tcb10 = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']

tick_font = 90 #70

# home_1D = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/'+\
#     'TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/IM_Residuals'
# ibaraki2011_srcmod_1d = home_1D+'/Results_ibaraki2011_srcmod_IM_residuals/'+\
#     'Flatfiles_IMs_ibaraki2011_srcmod_Residuals.csv'
# ibaraki2011_zheng1_1d = home_1D+'/Results_ibaraki2011_zheng1_IM_residuals/'+\
#     'Flatfiles_IMs_ibaraki2011_zheng1_Residuals.csv'
# iwate2011_zheng1_1d = home_1D+'/Results_iwate2011_zheng1_IM_residuals/'+\
#     'Flatfiles_IMs_iwate2011_zheng1_Residuals.csv'
# miyagi2011a_usgs_1d = home_1D+'/Results_miyagi2011a_usgs_IM_residuals/'+\
#     'Flatfiles_IMs_miyagi2011a_usgs_Residuals.csv'
# miyagi2011a_zheng1_1d = home_1D+'/Results_miyagi2011a_zheng1_IM_residuals/'+\
#     'Flatfiles_IMs_miyagi2011a_zheng1_Residuals.csv'
# nhonshu2012_zheng1_1d = home_1D+'/Results_n.honshu2012_zheng1_IM_residuals/'+\
#     'Flatfiles_IMs_n.honshu2012_zheng1_Residuals.csv'
# tokachi2003_srcmod1_1d = home_1D+'/Results_tokachi2003_srcmod1_IM_residuals/'+\
#     'Flatfiles_IMs_tokachi2003_srcmod1_Residuals.csv'
# tokachi2003_srcmod2_1d = home_1D+'/Results_tokachi2003_srcmod2_IM_residuals/'+\
#     'Flatfiles_IMs_tokachi2003_srcmod2_Residuals.csv'
# tokachi2003_srcmod3_1d = home_1D+'/Results_tokachi2003_srcmod3_IM_residuals/'+\
#     'Flatfiles_IMs_tokachi2003_srcmod3_Residuals.csv'
# tokachi2003_usgs_1d = home_1D+'/Results_tokachi2003_usgs_IM_residuals/'+\
#     'Flatfiles_IMs_tokachi2003_usgs_Residuals.csv'

# home_3D = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
#     '3D_Modeling_using_SW4/4_IM_Residuals_3D'
# ibaraki2011_srcmod_ptsrc1d = home_3D+'/Results_ibaraki2011_srcmod_ptsrc1d_IM_'+\
#     'residuals/Flatfiles_IMs_ibaraki2011_srcmod_ptsrc1d_Residuals.csv'
# ibaraki2011_srcmod_srf1d = home_3D+'/Results_ibaraki2011_srcmod_srf1d_IM_residuals/'+\
#     'Flatfiles_IMs_ibaraki2011_srcmod_srf1d_Residuals.csv'
# ibaraki2011_srcmod_3d = home_3D+'/Results_ibaraki2011_srcmod_srf3d_IM_'+\
#     'residuals/Flatfiles_IMs_ibaraki2011_srcmod_srf3d_Residuals.csv'
# ibaraki2011_zheng1_3d = home_3D+'/Results_ibaraki2011_zheng1_srf3d_IM_residuals/'+\
#     'Flatfiles_IMs_ibaraki2011_zheng1_srf3d_Residuals.csv'
# iwate2011_zheng1_3d = home_3D+'/Results_iwate2011_zheng1_srf3d_IM_residuals/'+\
#     'Flatfiles_IMs_iwate2011_zheng1_srf3d_Residuals.csv'
# miyagi2011a_usgs_3d = home_3D+'/Results_miyagi2011a_usgs_srf3d_IM_residuals/'+\
#     'Flatfiles_IMs_miyagi2011a_usgs_srf3d_Residuals.csv'
# miyagi2011a_usgs_z30km_3d = home_3D+'/Results_miyagi2011a_usgs_srf3d_z30km_valdivia_'+\
#     'IM_residuals/Flatfiles_IMs_miyagi2011a_usgs_srf3d_z30km_valdivia_Residuals.csv'
# miyagi2011a_zheng1_3d = home_3D+'/Results_miyagi2011a_zheng1_srf3d_IM_residuals/'+\
#     'Flatfiles_IMs_miyagi2011a_zheng1_srf3d_Residuals.csv'
# nhonshu2012_zheng1_3d = home_3D+'/Results_n.honshu2012_zheng1_srf3d_IM_residuals/'+\
#     'Flatfiles_IMs_n.honshu2012_zheng1_srf3d_Residuals.csv'
# tokachi2003_usgs_3d = home_3D+'/Results_tokachi2003_usgs_srf3d_IM_residuals/'+\
#     'Flatfiles_IMs_tokachi2003_usgs_srf3d_Residuals.csv'

# models = np.array([[ibaraki2011_srcmod_1d,ibaraki2011_srcmod_3d,'ibaraki2011_srcmod',[0,5]], #SNR_thresh
#                  [ibaraki2011_zheng1_1d,ibaraki2011_zheng1_3d,'ibaraki2011_zheng',[0,5]],
#                  [iwate2011_zheng1_1d,iwate2011_zheng1_3d,'iwate2011_zheng',[0,5]],
#                  [miyagi2011a_zheng1_1d,miyagi2011a_zheng1_3d,'miyagi2011a_zheng',[0,5]],
#                  [miyagi2011a_usgs_1d,miyagi2011a_usgs_3d,'miyagi2011a_usgs',[0,5]],
#                  [nhonshu2012_zheng1_1d,nhonshu2012_zheng1_3d,'nhonshu2012_zheng',[0,5]],
#                  [tokachi2003_usgs_1d,tokachi2003_usgs_3d,'tokachi2003_usgs',[0,5]]],dtype=object)


home_1D = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/'
iba_src_1d_025Hz = home_1D+'flatfile_1d_ibaraki2011_srcmod_0.25Hz_Residuals.csv'
iba_src_1d_050Hz = home_1D+'flatfile_1d_ibaraki2011_srcmod_0.49Hz_Residuals.csv'

iba_zh_1d_025Hz = home_1D+'flatfile_1d_ibaraki2011_zheng1_0.25Hz_Residuals.csv'
iba_zh_1d_050Hz = home_1D+'flatfile_1d_ibaraki2011_zheng1_0.49Hz_Residuals.csv'

iwa_zh_1d_025Hz = home_1D+'flatfile_1d_iwate2011_zheng1_0.25Hz_Residuals.csv'
iwa_zh_1d_050Hz = home_1D+'flatfile_1d_iwate2011_zheng1_0.49Hz_Residuals.csv'

miy_usgs_1d_025Hz = home_1D+'flatfile_1d_miyagi2011a_usgs_0.25Hz_Residuals.csv'
miy_usgs_1d_050Hz = home_1D+'flatfile_1d_miyagi2011a_usgs_0.49Hz_Residuals.csv'

miy_zh_1d_025Hz = home_1D+'flatfile_1d_miyagi2011a_zheng1_0.25Hz_Residuals.csv'
miy_zh_1d_050Hz = home_1D+'flatfile_1d_miyagi2011a_zheng1_0.49Hz_Residuals.csv'

tok_src1_1d_025Hz = home_1D+'flatfile_1d_tokachi2003_srcmod1_0.25Hz_Residuals.csv'
tok_src1_1d_050Hz = home_1D+'flatfile_1d_tokachi2003_srcmod1_0.49Hz_Residuals.csv'

tok_src2_1d_025Hz = home_1D+'flatfile_1d_tokachi2003_srcmod2_0.25Hz_Residuals.csv'
tok_src2_1d_050Hz = home_1D+'flatfile_1d_tokachi2003_srcmod2_0.49Hz_Residuals.csv'

tok_src3_1d_025Hz = home_1D+'flatfile_1d_tokachi2003_srcmod3_0.25Hz_Residuals.csv'
tok_src3_1d_050Hz = home_1D+'flatfile_1d_tokachi2003_srcmod3_0.49Hz_Residuals.csv'

tok_usgs_1d_025Hz = home_1D+'flatfile_1d_tokachi2003_usgs_0.25Hz_Residuals.csv'
tok_usgs_1d_050Hz = home_1D+'flatfile_1d_tokachi2003_usgs_0.49Hz_Residuals.csv'

#%% 3D 0.25Hz
home_3D_025Hz = "/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/3D_Modeling_using_SW4/Running_Simulations/4_Model_results/All_025Hz_Result/"

iba_src_3d_025Hz_las = home_3D_025Hz + 'flatfile_ibaraki2011_srcmod_srf3d_lasson_0.25Hz_Residuals.csv'
iba_src_3d_025Hz_tal = home_3D_025Hz + 'flatfile_ibaraki2011_srcmod_srf3d_talapas_0.25Hz_Residuals.csv'

iba_zh_3d_025Hz_las = home_3D_025Hz + 'flatfile_ibaraki2011_zheng1_srf3d_lasson_0.25Hz_Residuals.csv'
iba_zh_3d_025Hz_tal = home_3D_025Hz + 'flatfile_ibaraki2011_zheng1_srf3d_talapas_0.25Hz_Residuals.csv'

iwa_zh_3d_025Hz_las = home_3D_025Hz + 'flatfile_iwate2011_zheng1_srf3d_lasson_0.25Hz_Residuals.csv'
iwa_zh_3d_025Hz_tal = home_3D_025Hz + 'flatfile_iwate2011_zheng1_srf3d_talapas_0.25Hz_Residuals.csv'

miy_usgs_3d_025Hz_las = home_3D_025Hz + 'flatfile_miyagi2011a_usgs_srf3d_lasson_0.25Hz_Residuals.csv'
miy_usgs_3d_025Hz_tal = home_3D_025Hz + 'flatfile_miyagi2011a_usgs_srf3d_talapas_0.25Hz_Residuals.csv'

miy_zh_3d_025Hz_las = home_3D_025Hz + 'flatfile_miyagi2011a_zheng1_srf3d_lasson_0.25Hz_Residuals.csv'
miy_zh_3d_025Hz_tal = home_3D_025Hz + 'flatfile_miyagi2011a_zheng1_srf3d_talapas_0.25Hz_Residuals.csv'

miy_zh_3d_025Hz_30km = home_3D_025Hz + 'flatfile_miyagi2011a_zheng1_srf3d_z30km_talapas_0.25Hz_Residuals.csv'

tok_src3_3d_025Hz_tal = home_3D_025Hz + 'flatfile_tokachi2003_srcmod3_srf3d_talapas_0.25Hz_Residuals.csv'

tok_usgs_3d_025Hz_las = home_3D_025Hz + 'flatfile_tokachi2003_usgs_srf3d_lasson_0.25Hz_Residuals.csv'
tok_usgs_3d_025Hz_tal = home_3D_025Hz + 'flatfile_tokachi2003_usgs_srf3d_talapas_0.25Hz_Residuals.csv'



iba_src_3d_050Hz_tal = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/3D_Modeling_using_SW4/Running_Simulations/4_Model_results/All_05Hz_Results/flatfile_ibaraki2011_srcmod_srf3d_talapas_0.49Hz_Residuals.csv'

#%%

models = np.array([[iba_src_1d_025Hz, iba_src_3d_025Hz_tal, 'SW4_025_Hz',[0], [5]],
                   [iba_src_1d_050Hz, iba_src_3d_050Hz_tal, 'SW4_050_Hz',[0], [5]]],dtype=object)



att = ['pgd_res','tPGD_res','sd_res','xcorr'] #'tPGD_orig_res',
ls = ['-','--','-.','-','--','-.','-','--','-.','-','--']


for i in range(len(att)):
    gen_flatfile_kstest(models,str(att[i]))
    plot_pval_all(models,str(att[i]))
    plot_median_all(models,str(att[i]))

    plot_ks_test_all_w_median(str(att[i]))
    
# for legend
plot_median_all(models,str(att[0]),pltlegend=1)    
plot_ks_test_all_w_median(str(att[0]),pltlegend=1)


