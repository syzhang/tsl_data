"""All pain comparison as sanity check"""

import os,json,glob,sys
from os.path import join as opj
from nipype.interfaces.spm import Level1Design, EstimateModel, EstimateContrast, OneSampleTTestDesign, Threshold
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.fsl import Info
from nipype.algorithms.misc import Gunzip
from nipype import Workflow, Node
import nilearn.plotting
import numpy
import pandas as pd
import nibabel
import matplotlib.pyplot as plt

def first_level(TR):
    """define first level model"""
    # SpecifyModel - Generates SPM-specific Model
    modelspec = Node(SpecifySPMModel(concatenate_runs=False,
                                    input_units='secs',
                                    output_units='secs',
                                    time_repetition=TR,
                                    high_pass_filter_cutoff=128),
                                    name="modelspec")

    # Level1Design - Generates an SPM design matrix
    level1design = Node(Level1Design(bases={'hrf': {'derivs': [0, 0]}},
                                    timing_units='secs',
                                    interscan_interval=TR,
                                    model_serial_correlations='FAST'),
                                    name="level1design")

    # EstimateModel - estimate the parameters of the model
    level1estimate = Node(EstimateModel(estimation_method={'Classical': 1}),
                                    name="level1estimate")

    # EstimateContrast - estimates contrasts
    level1conest = Node(EstimateContrast(), name="level1conest")

    # Get Subject Info - get subject specific condition information
    getsubjectinfo = Node(Function(input_names=['subject_id'],
                                output_names=['subject_info'],
                                function=subjectinfo),
                        name='getsubjectinfo')

def subjectinfo(subject_id):
    """define individual subject info"""
    import pandas as pd
    from nipype.interfaces.base import Bunch
    
    def construct_sj(trialinfo, subject_id, run_num, cond_name):
        """construct df"""
        df_sj = trialinfo[(trialinfo['subject']==int(subject_id)) & (trialinfo['session']==int(run_num))]
        sj_info = pd.DataFrame()
        sj_info['onset'] = df_sj['runtime']
        sj_info['duration'] = 0.
        sj_info['weight'] = 1.
        trial_type = df_sj['seq'].replace({1:'Low', 2:'High'})
        sj_info['trial_type'] = trial_type
        sj_info_cond = sj_info[sj_info['trial_type']==cond_name]
        return sj_info_cond

    def select_confounds(subject_id, run_num, conf_names):
        """select confounds for regressor"""
        confounds_dir = f'/data/sub-%02d/func/' % int(subject_id)
        confounds_file = confounds_dir+f'sub-%02d_task-tsl_run-%d_desc-confounds_regressors.tsv' % (int(subject_id), int(run_num))
        conf_df = pd.read_csv(confounds_file, sep='\t')
        conf_select = conf_df[conf_names].loc[4:].fillna(0)
        conf_select_list = [conf_select[col].values.tolist() for col in conf_select] # ignore first 4 dummy scans
        return conf_select_list

    def find_runs(subject_id):
        """find available runs from func"""
        from glob import glob
        func_dir = f'/output/smooth_nomask/preproc/sub-%02d/' % int(subject_id)    
        func_files = glob(func_dir+'*bold.nii')
        runs = []
        for f in func_files:
            tmp = f.split('/')
            run = tmp[5].split('_')[2].split('-')[1]
            runs.append(int(run))
        return sorted(runs)
    
    conf_names = ['csf','white_matter','global_signal',
    'std_dvars','dvars', 'framewise_displacement', 'rmsd',
    'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03',
    'a_comp_cor_04', 'a_comp_cor_05', 
    'trans_x', 'trans_y', 'trans_z', 'rot_x','rot_y','rot_z']
#     't_comp_cor_00', 't_comp_cor_01', 't_comp_cor_02',
                      #'motion_outlier00', 'motion_outlier01','motion_outlier02', 'motion_outlier03']

    alltrialinfo = pd.read_csv('/tsl_data/data/fmri_behavioural.csv')
    alltrialinfo.head()
    
    subject_info = []
    onset_list = []
    condition_names = ['High', 'Low']
    runs = find_runs(subject_id)
    print(runs)
    for run in runs:
        for cond in condition_names:
            run_cond = construct_sj(alltrialinfo, subject_id, run, cond)
            onset_run_cond = run_cond['onset'].values
            onset_list.append(sorted(onset_run_cond))

    subject_info = []
    for r in range(len(runs)):
        onsets = [onset_list[r*2], onset_list[r*2+1]]
        regressors = select_confounds(subject_id, runs[r], conf_names)

        subject_info.insert(r,
                           Bunch(conditions=condition_names,
                                 onsets=onsets,
                                 durations=[[0], [0]],
                                 regressors=regressors,
                                 regressor_names=conf_names,
                                 amplitudes=None,
                                 tmod=None,
                                 pmod=None))

    return subject_info  # this output will later be returned to infosource

def select_confounds(subject_id, run_num):
    """import confounds tsv files"""
    confounds_dir = f'/data/sub-%02d/func/' % int(subject_id)
    confounds_file = confounds_dir+f'sub-%02d_task-tsl_run-%d_desc-confounds_timeseries.tsv' % (int(subject_id), int(run_num))
    conf_df = pd.read_csv(confounds_file, sep='\t')
    return conf_df

def confounds_regressor(conf_df, conf_names):
    """select confounds for regressors"""
    conf_select = conf_df[conf_names].loc[4:].fillna(0)
    conf_select_list = [conf_select[col].values.tolist() for col in conf_select] # ignore first 4 dummy scans
    return conf_select_list

def list_subject(data_dir='/data'):
    """list all available subjects"""
    sj_ls = []
    for f in os.listdir(data_dir):
        if f.startswith('sub') and (not f.endswith('.html')):
            sj_ls.append(f.split('-')[1])
    return sj_ls

def find_runs(subject_id):
    """find available runs from func"""
    from glob import glob
    func_dir = f'/output/smooth_nomask/preproc/sub-%02d/' % int(subject_id)    
    func_files = glob(func_dir+'*bold.nii')
    runs = []
    for f in func_files:
        tmp = f.split('/')
        run = tmp[5].split('_')[2].split('-')[1]
        runs.append(int(run))
    return sorted(runs)


if __name__ == "__main__":
    experiment_dir = '/output'
    output_dir = 'smooth_nomask'
    working_dir = 'workingdir'
    data_dir = '/data'
    code_dir = '/code'

    alltrialinfo = pd.read_csv('/code/data/fmri_behavioural_new.csv')
    for sj in list_subject(data_dir=data_dir):
    # for sj in ['06']:
        runs = find_runs(sj)
        for run in runs:
            df_tmp = select_confounds(sj, run)
            tmp = df_tmp.filter(regex='outlier')
            print(f'subject %s (run %s) confounds shape %d x %d' % (str(sj), str(run), tmp.shape[0], tmp.shape[1]))