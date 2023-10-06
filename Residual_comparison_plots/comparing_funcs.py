#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:32:34 2023

@author: oluwaseunfadugba
"""

#%% plot_compare_IM_res
def plot_compare_IM_res(axes,data,x_axis="hypdist_index",y_axis="pgd_res",n_rupt='all',
                        xticks=None,yticks=None,subplt_label=None,subplt_labelpos=[-0.2, 1.1],
                        title=None,title_pad = 0, tag='',ylim=[0,1],xlabel=None, #'distance (km)' "PGD residual"
                        ylabel='ln Residual',fontsize=130,pltlegend=0,opt_SNR=0,SNR_threshold=999):

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.patches import Patch
    import os
    
    sns.set_theme(style="ticks", color_codes=True)

    axes.set_title(title,fontsize=fontsize+15,fontdict={"weight": "bold"},pad =title_pad)
    
    flatfile_res_paths = data[:,0]
    colors = data[:,1]
    legend = data[:,2]   
    hatches = data[:,3]
    boxes = [0]
    zeroline_thick = 20
    
    for i in range(len(flatfile_res_paths)):
        #print(flatfile_res_paths[i])
        globals()['flatfile_res_dataframe_%s' % i] = pd.read_csv(flatfile_res_paths[i]) 
        
        if n_rupt != 'all':
            globals()['flatfile_res_dataframe_%s' % i] = \
                globals()['flatfile_res_dataframe_%s' % i][globals()['flatfile_res_dataframe_%s' % i]
                                                           ['rupt_no'].isin(n_rupt)]
            
        if opt_SNR == 1:
            globals()['flatfile_res_dataframe_%s' % i] = \
                globals()['flatfile_res_dataframe_%s' % i][globals()['flatfile_res_dataframe_%s' % i]
                                                           ['obs_SNR']>=SNR_threshold]
            
            if len(globals()['flatfile_res_dataframe_%s' % i])==0:
                return
        
        sns.boxplot(ax=axes,x=x_axis, y=y_axis, data=globals()['flatfile_res_dataframe_%s' % i],
                    color=colors[i],boxprops=dict(alpha=.6,edgecolor='k',linewidth=10),
                    capprops=dict(alpha=.6,color=colors[i], linewidth=10),
                    whiskerprops=dict(alpha=.7,color=colors[i],ls = '--', linewidth= 10),
                    medianprops={"linewidth":10},showfliers=False,linewidth=7) #"alpha":.6,"color":'k'

        #printing hatches
        boxes.append(len(axes.patches))
        for j, patch in enumerate(axes.patches):
            if j <= boxes[i+1] and j >= boxes[i]:
                patch.set_hatch(hatches[i])
        
    # Increasing the linewidth of the frame border 
    for pos in ['right', 'top', 'bottom', 'left']:
        axes.spines[pos].set_linewidth(zeroline_thick/2)
    
    axes.axhline(y=0, ls='-',linewidth=zeroline_thick, color='r')
    axes.grid(axis = 'y',color = 'k', linestyle = '--', linewidth = zeroline_thick/3)
    axes.set(ylim=(ylim[0], ylim[1]))
    axes.tick_params(axis='x',labelsize=fontsize,labelrotation=45,length=20, width=10)
    axes.tick_params(axis='y',labelsize=fontsize,labelrotation=0,length=20, width=10)
    axes.set_xlabel(xlabel, fontsize=fontsize)
    axes.set_ylabel(ylabel, fontsize=fontsize)

    if xticks != None:
        # get the x-tick positions and labels
        xticklabels = axes.get_xticklabels()
        xticklabels2 = [label.get_text() for label in xticklabels]

        for tic in range(len(xticklabels2)):
            if int(xticklabels2[tic]) not in xticks:
                xticklabels2[tic] = ''

        # set the x-tick positions and labels with actual values
        axes.set_xticklabels(xticklabels2)
    
    if subplt_label != None:
        # Add alphabet labels to the subplots
        axes.text(subplt_labelpos[0], subplt_labelpos[1], '('+str(subplt_label)+')', 
                  transform=axes.transAxes, fontsize=fontsize, fontweight="bold", va="top")



    if pltlegend == 1:
        # Creating legend
        legend_elements = []
        for i in range(len(flatfile_res_paths)):
            legend_elements.append(Patch(alpha=0.7,edgecolor='black',linewidth=5))
        
        for j, patch in enumerate(legend_elements):
            patch.set_hatch(hatches[j])
            patch.set_label(legend[j])
            patch.set_facecolor(colors[j])
            patch.set_alpha(0.6)
            
        axes.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
                    fontsize=fontsize-10,frameon=False)
    #     figpath = os.getcwd() +'/fig.'+tag+'.legend.IM.png'
    # else:

    #     figpath = os.getcwd() +'/fig.'+tag+'.IM.png'
    
    #plt.savefig(figpath, bbox_inches='tight', dpi=100)
    #plt.close()  
    #plt.show()
    return


#%% plot_median_residual
def plot_median_residual(axes,models,attribute='pgd_res',title=None,title_pad = 0,
                         pltlegend = 0,xlabel=None,ylabel = 'ln',pvalue = 0,
                         kstest_ff_path='',fontsize=130,xticks=None,yticks=None,
                         subplt_label=None,subplt_labelpos=[-0.2, 1.1],
                         draw_rec = 1,rec_neg=1,rec_color='gray',rec_alpha = 0.15):
                    
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    #import os
    import matplotlib.patches as patches
    
    flatfile = kstest_ff_path+"flatfile_ks_test_"+attribute+".csv"

    factor = 4
    lw = 15
    zeroline_thick = 20
    
    ls = ['-','--','-.','-','--','-.','-','--','-.','-','--']
    
    # fig, axes = plt.subplots(figsize=(30, 20))
    axes.set_title(title,fontsize=fontsize+15,fontdict={"weight": "bold"},pad =title_pad)
    
    
    for i in range(len(models[:,0])):       
        df = pd.read_csv(flatfile) 
        df = df[(df['tag']==str(models[i,2])) & (df['ks_hypo']!=0) & 
                (df['SNR_thresh']==0)].reset_index(drop=True)
    
        if pvalue == 0:
            axes.plot(df['ks_hypo'],np.abs(df['ks_median_y'])-np.abs(df['ks_median_x']),
                      lw=lw,ls = ls[i],label=str(models[i,2]))
            axes.scatter(df['ks_hypo'],np.abs(df['ks_median_y'])-np.abs(df['ks_median_x']),
                        s = df['ks_n']*factor)
        else:
            
            axes.plot(df['ks_hypo'],df['ks_pval'],lw=lw,ls = ls[i],label=str(models[i,2]))
            axes.scatter(df['ks_hypo'],df['ks_pval'],s = df['ks_n']*factor)

    axes.axhline(y=0, c= 'k',ls='--',linewidth=zeroline_thick,label='3D-1D Res = 0')    
    axes.tick_params(axis='x',labelsize=fontsize,labelrotation=45,length=20, width=10)
    if xticks != None:
        axes.set_xticks(ticks=xticks)
    if yticks != None:
        axes.set_yticks(ticks=yticks)
    axes.tick_params(axis='y',labelsize=fontsize,labelrotation=0,length=20, width=10)
    axes.set_xlabel(xlabel, fontsize=fontsize)
    
    if pvalue == 0:
        axes.set_ylabel('$\delta_{|3D|-|1D|} $ (' + ylabel+')', fontsize=fontsize)
    else:
        axes.set_ylabel('p-value', fontsize=fontsize)
    
    if draw_rec == 1:
        xmin,xmax = axes.get_xlim()
        ymin,ymax = axes.get_ylim()
        
        if rec_neg == 1:
            r1 = patches.Rectangle((0,0), xmax, ymin, color=rec_color, alpha=rec_alpha)
            axes.add_patch(r1)
        else:
            r1 = patches.Rectangle((0,0), xmax, ymax, color=rec_color, alpha=rec_alpha)
            axes.add_patch(r1)
    
    if subplt_label != None:
        # Add alphabet labels to the subplots
        axes.text(subplt_labelpos[0], subplt_labelpos[1], '('+str(subplt_label)+')', 
                  transform=axes.transAxes, fontsize=fontsize, fontweight="bold", va="top")
        
    if pltlegend == 1:
        axes.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0,fontsize=fontsize)
    #     figpath = os.getcwd() +'/fig.kstest_median_w_dist_'+attribute+'.legend.IM_residuals.png'
    # else:
    #     figpath = os.getcwd() +'/fig.kstest_median_w_dist_'+attribute+'.IM_residuals.png'
    
    axes.grid()
    
    #axes.set(xlim=(0, 1100)) #1600
    
    axes.grid(color = 'k', linestyle = '-', linewidth = 2)
    [i.set_linewidth(4) for i in axes.spines.values()]

    #plt.savefig(figpath, bbox_inches='tight', dpi=100)
    
    #plt.close()  
    
    
    
    
    
    
#%% plot_ks_test_all_w_median
def plot_ks_test_all_w_median(axes,attribute='pgd_res',title=None,title_pad = 0,
                         pltlegend = 0,xlabel=None,ylabel = 'ln',pvalue = 0,
                         kstest_ff_path='',fontsize=130,xticks=None,yticks=None,
                         subplt_label=None,subplt_labelpos=[-0.2, 1.1],lw = 12,
                         draw_rec = 1,rec_neg=1,rec_color='gray',rec_alpha = 0.15):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker


    tcb10 = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', 
             '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']


    flatfile = kstest_ff_path+"flatfile_ks_test_"+attribute+".csv"
    
    df = pd.read_csv(flatfile) 
    df = df[(df['ks_hypo']==0) & (df['SNR_thresh']==0)].reset_index(drop=True)

    axes_max = (max(np.abs( np.concatenate((df['ks_stat'].to_numpy(),
                                            df['ks_Dcrit'].to_numpy(),
                                            df['ks_pval'].to_numpy()))))*1)/1
    factor = 1.5
    
    #plt.style.use('tableau-colorblind10') #('seaborn-colorblind') #
    axes.set_title(title,fontsize=fontsize+15,fontdict={"weight": "bold"},pad =title_pad)
    
    axes.plot(df['tag'],df['ks_stat'],c=tcb10[0],ls='-',lw=lw,label='ks-stat') #bo
    axes.scatter(df['tag'],df['ks_stat'],c=tcb10[0],s = df['ks_n']*factor) #c='b',
    axes.plot(df['tag'],df['ks_Dcrit'],c=tcb10[0],ls='-.',lw=lw,label='D Critical') #,ms = 20 bo
    
    axes.plot(df['tag'],df['ks_pval'],c=tcb10[1],ls='-',lw=lw,label='p value') #ro
    axes.scatter(df['tag'],df['ks_pval'],c=tcb10[1],s = df['ks_n']*factor) #c='r'
    axes.axhline(y=0.05, c='#FF800E',ls='--',linewidth=lw,label='pval = 0.05') #c= 'r',
    
    #print(df['tag'][0])
    axes.tick_params(axis='x',labelsize=fontsize)#,labelrotation=60,length=20, width=3)
    plt.setp(axes.get_xticklabels(),rotation=45, ha="right",rotation_mode="anchor")
    
    if xticks.any():
        # get the x-tick positions and labels
        xticklabels = axes.get_xticklabels()
        xticklabels2 = [label.get_text() for label in xticklabels]

        for tic in range(len(df['tag'])):
            if str(df['tag'][tic]) == str(xticks[0,0]):
                xticklabels2[tic] = xticks[0,1]
            elif str(df['tag'][tic]) == str(xticks[1,0]):
                xticklabels2[tic] = xticks[1,1]
            elif str(df['tag'][tic]) == str(xticks[2,0]):
                xticklabels2[tic] = xticks[2,1]
            elif str(df['tag'][tic]) == str(xticks[3,0]):
                xticklabels2[tic] = xticks[3,1]
            elif str(df['tag'][tic]) == str(xticks[4,0]):
                xticklabels2[tic] = xticks[4,1]
            elif str(df['tag'][tic]) == str(xticks[5,0]):
                xticklabels2[tic] = xticks[5,1]
            elif str(df['tag'][tic]) == str(xticks[6,0]):
                xticklabels2[tic] = xticks[6,1]

        # set the x-tick positions and labels with actual values
        axes.set_xticklabels(xticklabels2)
    
    
    axes.tick_params(axis='y',labelsize=fontsize,labelrotation=0,length=20, width=3)
    
    axes.set_xlabel(xlabel, fontsize=fontsize)
    axes.set(ylim=(-axes_max-(axes_max/10), axes_max+(axes_max/10)))
    
    [i.set_linewidth(4) for i in axes.spines.values()]
    
    #axes.set_ylabel('k-s test/p-value', fontsize=fontsize)
    ybox1 = TextArea("ks-test ", textprops=dict(color=tcb10[0], size=fontsize,rotation=90,ha='left',va='bottom'))
    ybox2 = TextArea("/",     textprops=dict(color="k", size=fontsize,rotation=90,ha='left',va='bottom'))
    ybox3 = TextArea("p-value ", textprops=dict(color=tcb10[1], size=fontsize,rotation=90,ha='left',va='bottom'))
    
    ybox = VPacker(children=[ybox3, ybox2, ybox1],align="bottom", pad=0, sep=5)
    
    anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False, bbox_to_anchor=(-0.15, 0.2), 
                                      bbox_transform=axes.transAxes, borderpad=0.)
    
    axes.add_artist(anchored_ybox)
    
    if subplt_label != None:
        # Add alphabet labels to the subplots
        axes.text(subplt_labelpos[0], subplt_labelpos[1], '('+str(subplt_label)+')', 
                  transform=axes.transAxes, fontsize=fontsize, fontweight="bold", va="top")
        
    
    #----------------------------------------------
    
    axes2=axes.twinx()
    
    axes2.plot(df['tag'],np.abs(df['ks_median_y'])-np.abs(df['ks_median_x']),
               c=tcb10[2],ls='-',lw=lw,label='med_3D-1D') #go
    axes2.scatter(df['tag'],np.abs(df['ks_median_y'])-np.abs(df['ks_median_x']),
                  c=tcb10[2],s = df['ks_n']*1.5) #,c='g'
    
    axes2.set_ylabel('$\delta_{|3D|-|1D|}$ ('+ylabel+')', c=tcb10[2],fontsize=fontsize)
    axes2.tick_params(axis='y',colors=tcb10[2],labelsize=fontsize,labelrotation=0,length=20, width=3)
    
    axes2_max = (max(np.abs(np.abs(df['ks_median_y'])-np.abs(df['ks_median_x'])))*1)/1
    axes2.set(ylim=(-axes2_max-(axes2_max/10), axes2_max+(axes2_max/10)))
    axes2.spines['right'].set_color(tcb10[2])
    

    import matplotlib.ticker
    nticks = 5
    axes.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nticks)) #LinearLocator
    axes2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nticks))

    if pltlegend == 1:
        axes2.scatter(9.1, -axes2_max+(axes2_max/2), s=round(max(df['ks_n']*factor)), c=tcb10[0])
        axes2.scatter(9.1, -axes2_max+(axes2_max/4), s=round(min(df['ks_n']*factor)), c=tcb10[0])
   
        axes.legend(bbox_to_anchor=(1.25, 1), loc='upper left', borderaxespad=0,fontsize=fontsize,frameon=False)
      
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
                     loc='upper left',fontsize=fontsize,frameon=False)

        #figpath = os.getcwd() +'/fig.kstest_all_median_'+attribute+'.legend.IM_residuals.png'
    
    else:
        axes.grid(color = 'k', linestyle = '-', linewidth = 2)
        axes2.grid(None)
        
        #figpath = os.getcwd() +'/fig.kstest_all_median_'+attribute+'.IM_residuals.png'
    
    #plt.savefig(figpath, bbox_inches='tight', dpi=100)

    #plt.close()  

    return
   