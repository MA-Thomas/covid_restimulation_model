#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 16:43:16 2022

@author: marcus
"""
from hyperopt import hp
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import space_eval
from hyperopt import STATUS_OK

import multiprocessing
from joblib import Parallel, delayed
#from tqdm import tqdm
import concurrent.futures

import argparse
import json
import os
from collections import defaultdict
import glob

import numpy as np
import pandas as pd

import time
from scipy.stats import entropy
from scipy.spatial.distance import hamming

import pickle
import sys

sys.path.insert(0,'/mnt/scratch/marcust/WorkingDir/CFIT')
from cfit.util.Utils import Utils
from cfit.util.PeptideDistance import PeptideDistance

# # NeoPipe and netMHC40 section.
# sys.path.insert(1, '/mnt/data/software/NeoPipe') # # Either works
sys.path.insert(0, '/mnt/scratch/marcust/WorkingDir/NeoPipe')

from neopipe.NetMHC40 import NetMHC40
os.system('chmod +x /mnt/data/software/netMHC-4.0/netMHC')
if "/mnt/data/software/netMHC-4.0" not in os.environ["PATH"]:
    os.environ["PATH"] = "/mnt/data/software/netMHC-4.0"+ os.pathsep + os.environ["PATH"]
else:
    print("already there")


import math
import random
# # # The explicit class import is needed to load the pickle file without error.
# # # See https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
# import vptree
# from vptree import VPTree

sys.path.append('/mnt/scratch/marcust/WorkingDir/CFIT')
from cfit.fitness.neo_quality.EpitopeDistance import EpitopeDistance

import matplotlib.pyplot as plt
import statistics as stats
from scipy.stats.mstats import gmean

# # for plot_regr()
# import matplotlib.pyplot as plt
# import numpy as np
from scipy.stats import pearsonr, linregress
from scipy.stats import spearmanr
from scipy.stats import kendalltau

import matplotlib as mpl




HLAA_colon = ['HLA-A01:01', 'HLA-A02:01', 'HLA-A02:02', 'HLA-A02:03', 'HLA-A02:05', 'HLA-A02:06', 'HLA-A02:07', 'HLA-A02:11',
        'HLA-A02:12', 'HLA-A02:16', 'HLA-A02:17', 'HLA-A02:19', 'HLA-A02:50', 'HLA-A03:01', 'HLA-A03:02', 'HLA-A03:19',
        'HLA-A11:01', 'HLA-A23:01', 'HLA-A24:02', 'HLA-A24:03', 'HLA-A25:01', 'HLA-A26:01', 'HLA-A26:02', 'HLA-A26:03',
        'HLA-A29:02', 'HLA-A30:01', 'HLA-A30:02', 'HLA-A31:01', 'HLA-A32:01', 'HLA-A32:07', 'HLA-A32:15', 'HLA-A33:01',
        'HLA-A66:01', 'HLA-A68:01', 'HLA-A68:02', 'HLA-A68:23', 'HLA-A69:01', 'HLA-A80:01']

HLAB_colon = ['HLA-B07:02', 'HLA-B08:01', 'HLA-B08:02', 'HLA-B0803', 'HLA-B14:01', 'HLA-B14:02', 'HLA-B15:01', 'HLA-B15:02',
        'HLA-B15:03', 'HLA-B15:09', 'HLA-B15:17', 'HLA-B18:01', 'HLA-B27:05', 'HLA-B27:20', 'HLA-B35:01', 'HLA-B35:03',
        'HLA-B37:01', 'HLA-B38:01', 'HLA-B39:01', 'HLA-B40:01', 'HLA-B40:02', 'HLA-B40:13', 'HLA-B42:01', 'HLA-B44:02',
        'HLA-B44:03', 'HLA-B45:01', 'HLA-B45:06', 'HLA-B46:01', 'HLA-B48:01', 'HLA-B51:01', 'HLA-B53:01', 'HLA-B54:01',
        'HLA-B57:01', 'HLA-B57:03', 'HLA-B58:01', 'HLA-B58:02', 'HLA-B73:01', 'HLA-B81:01', 'HLA-B83:01']

HLAC_colon = ['HLA-C03:03', 'HLA-C04:01', 'HLA-C05:01', 'HLA-C06:02', 'HLA-C07:01', 'HLA-C07:02', 'HLA-C08:02', 'HLA-C12:03',
        'HLA-C14:02', 'HLA-C15:02']
HLA_with_colon = HLAA_colon+HLAB_colon+HLAC_colon


#all hla types handled by netMHC4.0
HLAA = ['HLA-A0101', 'HLA-A0201', 'HLA-A0202', 'HLA-A0203', 'HLA-A0205', 'HLA-A0206', 'HLA-A0207', 'HLA-A0211',
        'HLA-A0212', 'HLA-A0216', 'HLA-A0217', 'HLA-A0219', 'HLA-A0250', 'HLA-A0301', 'HLA-A0302', 'HLA-A0319',
        'HLA-A1101', 'HLA-A2301', 'HLA-A2402', 'HLA-A2403', 'HLA-A2501', 'HLA-A2601', 'HLA-A2602', 'HLA-A2603',
        'HLA-A2902', 'HLA-A3001', 'HLA-A3002', 'HLA-A3101', 'HLA-A3201', 'HLA-A3207', 'HLA-A3215', 'HLA-A3301',
        'HLA-A6601', 'HLA-A6801', 'HLA-A6802', 'HLA-A6823', 'HLA-A6901', 'HLA-A8001']

HLAB = ['HLA-B0702', 'HLA-B0801', 'HLA-B0802', 'HLA-B0803', 'HLA-B1401', 'HLA-B1402', 'HLA-B1501', 'HLA-B1502',
        'HLA-B1503', 'HLA-B1509', 'HLA-B1517', 'HLA-B1801', 'HLA-B2705', 'HLA-B2720', 'HLA-B3501', 'HLA-B3503',
        'HLA-B3701', 'HLA-B3801', 'HLA-B3901', 'HLA-B4001', 'HLA-B4002', 'HLA-B4013', 'HLA-B4201', 'HLA-B4402',
        'HLA-B4403', 'HLA-B4501', 'HLA-B4506', 'HLA-B4601', 'HLA-B4801', 'HLA-B5101', 'HLA-B5301', 'HLA-B5401',
        'HLA-B5701', 'HLA-B5703', 'HLA-B5801', 'HLA-B5802', 'HLA-B7301', 'HLA-B8101', 'HLA-B8301']

HLAC = ['HLA-C0303', 'HLA-C0401', 'HLA-C0501', 'HLA-C0602', 'HLA-C0701', 'HLA-C0702', 'HLA-C0802', 'HLA-C1203',
        'HLA-C1402', 'HLA-C1502']

HLA = HLAA + HLAB + HLAC
WUHAN_Spike = 'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'

distFun = 'all_tcr_all_combos_model'
epidist = EpitopeDistance(model_name=distFun)
def dist(point1, point2):
    d = epidist.epitope_dist(point1[1], point2[1])
    return d



def LSE(element_list):
    max_x = max(element_list)
    LSE = max_x + np.log( sum( [ np.exp(x-max_x) for x in element_list] ) )

    return LSE

def realSoftMax(element_list, a_soft, func='Boltz'):

    if func=='Boltz':
        '''
        Implements the Boltzmann operator as the smooth approx to max()
        '''
        element_list = [x for x in element_list if x > 0]

        logRSMnumerator = Utils.log_sum(a_soft*np.array(element_list) + np.log(element_list))
        logRSMdenomenator = Utils.log_sum(a_soft*np.array(element_list))
        Out = np.exp( logRSMnumerator - logRSMdenomenator )


    elif func=='Mellow':
        '''
        Implements the Mellowmax as the smooth approx to max()
        See Asadi, Kavosh; Littman, Michael L. (2017). "An Alternative Softmax Operator for Reinforcement Learning". PMLR. 70: 243–252. Retrieved January 6, 2023.
        '''
        if a_soft == 0:
            a_soft = 1e-18 # The limit as a_soft approches 0 is valid (Out approaches the mean()).

        expTerm = np.exp( Utils.log_sum(a_soft*np.array(element_list)) )
        n = len(element_list)
        Out = np.log((1/n)*expTerm) / a_soft

    return Out

def realSoftMin(element_list, a_soft, func='Boltz'):

    if func=='Boltz':
        '''
        Implements the Boltzmann operator as the smooth approx to min()
        '''
        element_list = [x for x in element_list if x > 0]

        logRSMnumerator = Utils.log_sum(-a_soft*np.array(element_list) + np.log(element_list))
        logRSMdenomenator = Utils.log_sum(-a_soft*np.array(element_list))
        Out = np.exp( logRSMnumerator - logRSMdenomenator )


    elif func=='Mellow':
        '''
        Implements the Mellowmin as the smooth approx to min()
        See Asadi, Kavosh; Littman, Michael L. (2017). "An Alternative Softmax Operator for Reinforcement Learning". PMLR. 70: 243–252. Retrieved January 6, 2023.
        '''
        if a_soft == 0:
            a_soft = 1e-18 # The limit as a_soft approches 0 is valid (Out approaches the mean()).

        expTerm = np.exp( Utils.log_sum(-a_soft*np.array(element_list)) )
        n = len(element_list)
        Out = np.log((1/n)*expTerm) / -a_soft

    return Out

def generate_seq_hla_Info_SPIKE(dfs):
    # # Look at Peptides
    hla_seqq_dict = {}

    peptide_df = dfs['Peptide sequences for mutations']

    # # Get Cansu sequences
    long_mt_sequences = list(set(peptide_df['Mutant Sequence']))
    long_wt_sequences = list(set(peptide_df['WT Sequence ']))

    hla_df = dfs['HLAseq of donors']
    hla_class1_matrix = hla_df.iloc[1:14,0:7]

    patientID_to_seq_hla_dict = {}
    for i,patient_row in hla_class1_matrix.iterrows():

        patient_id = patient_row[0]
        if patient_id not in patientID_to_seq_hla_dict:
            patientID_to_seq_hla_dict[patient_id] = []

        # # Cansu's HLA formatting is different. Make appropriate changes
        # # to netMHC40 format.
        patient_HLAs = list(patient_row[1:])
        patient_HLAs = convertCansuHLAformat(patient_HLAs)

        # # Iterate over patient's hlas.
        for hla in patient_HLAs:
            # # Fill patientID_to_seq_hla_dict.
            for seq in long_mt_sequences:
                if [seq,hla] not in patientID_to_seq_hla_dict[patient_id]:
                    patientID_to_seq_hla_dict[patient_id].append([seq,hla])
                    print("mt (seq,hla): ",seq,hla, flush=True)

            for seq in long_wt_sequences:
                if [seq,hla] not in patientID_to_seq_hla_dict[patient_id]:
                    patientID_to_seq_hla_dict[patient_id].append([seq,hla])
                    print("wt (seq,hla): ",seq,hla, flush=True)


    seqq_list = [] # Used to fill hla_seqq_dict
    for seq in long_wt_sequences+long_mt_sequences:
        seqq_list.extend( generate_9_mers(seq) )
    seqq_list = list( set(seqq_list))

    # # Add WUHAN_Spike 9mers.
    seqq_list.extend( list(set( generate_9_mers(WUHAN_Spike))))
    seqq_list = list( set(seqq_list))

    # # Iterate over ALL hlas.
    for hla in HLA:
        hla_seqq_dict[hla] = seqq_list
    return patientID_to_seq_hla_dict, hla_seqq_dict

def generate_9_mers(seq):
    list_9mers = []
    for i in range(len(seq)-8):
        list_9mers.append(seq[i:i+9])

    return list_9mers

def convertCansuHLAformat(hla_set, shuffle_HLAs=False):
    # # Cansu's HLA formatting is different. Make appropriate changes
    # # to netMHC40 format with no colons.
    hlaAs = ['HLA-A'+item[:5] for item in hla_set[:2]]
    hlaBs = ['HLA-B'+item[:5] for item in hla_set[2:4]]
    hlaCs = ['HLA-C'+item[:5] for item in hla_set[4:6]]
    hla_set = hlaAs+hlaBs+hlaCs
    hla_set = [item.replace(":","") for item in hla_set if item in HLA_with_colon]
    hla_set = list(set(hla_set))


    return hla_set

def generatePatientResponseInfo_SPIKE(dfs, patientID_to_seq_hla_dict, CD8_response_str, pepID_EXCEL_COL):
    '''
    Construct the dictionary ResponseInfo[patient][Protein Pool][PeptideID] = resp
    For spike, the protein pool is WT vs MT
    '''
    ResponseInfo = {}
    PatientID_PepID_Dict = {}
    PeptideID_to_Sequence_Dict = {}

    # # Look at Peptides
    peptide_df = dfs['Peptide sequences for mutations']

    cd8_responses = dfs[CD8_response_str]
    patient_listing = ['Unnamed: 3', 'Unnamed: 5', 'Unnamed: 7', 'Unnamed: 9', 'Unnamed: 11', 'Unnamed: 13',
                       'Unnamed: 15', 'Unnamed: 17', 'Unnamed: 19', 'Unnamed: 21', 'Unnamed: 23', 'Unnamed: 25']
    patient_listing_numbers = [8001, 8009, 8052, 9001, 9003, 9005,
                               9006, 9008, 9012, 9017, 9019, 9020]
    IDs_Sequences_MT = peptide_df[['Peptide ID','Mutant Sequence']]
    IDs_Sequences_WT = peptide_df[['Peptide ID','WT Sequence ']]

    for patient_id, seq_hla_list in patientID_to_seq_hla_dict.items():
        print("patient_id: ", patient_id)
        # # Cansu's data is missing CD8+/CD4+ responses for patient 9004
        if patient_id == 9004:
            continue

        ResponseInfo[patient_id] = {}
        ResponseInfo[patient_id]['MT'] = {}
        ResponseInfo[patient_id]['WT'] = {}
        PatientID_PepID_Dict[patient_id] = []

        for (seq,hla) in seq_hla_list:
            print("for loop - seq,hla: ", seq, hla)
            # # There may exist multiple copies of the same WT each with diff
            # # MTs. If seq is one of these WTs with multiple copies, we need
            # # to handle each - each will have a different peptideID.
            # # Using this peptideID_list variable in when seq is a MT is just
            # # to make the code general.
            peptideID_list = []

            # # Get Peptide ID of current sequence
            # # First consider that it may be an mt sequence
            if seq in list(IDs_Sequences_MT['Mutant Sequence']):
                print("mt seq: ",seq, flush=True)
                peptideID = list(IDs_Sequences_MT.loc[IDs_Sequences_MT['Mutant Sequence'] == seq]['Peptide ID'])[0]
                peptideID = str(peptideID)+'MT'
                PeptideID_to_Sequence_Dict[peptideID] = seq
                peptideID_list.append(peptideID)
                is_mt = True
                is_wt = False

            # # Next consider if it is instead a wt sequence.
            # # There may me more than 1 WT sequence in the Spike data.
            # # Each will have a diff MT.
            else:
                peptideIDs = list(IDs_Sequences_WT.loc[IDs_Sequences_WT['WT Sequence '] == seq]['Peptide ID'])
                print("WT peptideIDs: ", peptideIDs)
                if len(peptideIDs) == 1:
                    peptideID = peptideIDs[0]
                    peptideID = str(peptideID)+'WT'
                    print("if case. wt seq/peptideID: ",seq, peptideID, flush=True)
                    PeptideID_to_Sequence_Dict[peptideID] = seq
                    peptideID_list.append(peptideID)
                elif len(peptideIDs) > 1:
                    for peptideID in peptideIDs:
                        peptideID = str(peptideID)+'WT'
                        print("else case. wt seq/peptideID: ",seq, peptideID, flush=True)
                        PeptideID_to_Sequence_Dict[peptideID] = seq
                        peptideID_list.append(peptideID)
                is_mt = False
                is_wt = True

            # # Store CD8 response for current patient at current sequence
            for peptideID in peptideID_list:
                print("peptideID: ", peptideID)
                if peptideID not in ResponseInfo[patient_id]['MT'] and peptideID not in ResponseInfo[patient_id]['WT']:

                    print("Cansu peptideID: ", peptideID, flush=True)

                    # # Get patient column, mutant sequences have column names like Unnamed: 3
                    # # wt sequences have column names like 8001
                    if is_mt:
                        patientCOL = [patient_listing[i] for i in range(len(patient_listing)) if patient_listing_numbers[i] == patient_id][0]
                        print("Cansu patientCOL (mt seq): ",patientCOL, flush=True)
                        pID = int(peptideID.replace('MT',''))
                        resp = list(cd8_responses.loc[cd8_responses[pepID_EXCEL_COL] == pID][patientCOL])[0]
                        if math.isnan( resp ):
                            # del ResponseInfo[patient_id]['MT'][peptideID]
                            continue
                        else:
                            ResponseInfo[patient_id]['MT'][peptideID] = resp
                        print("Current seq, ResponseInfo[patient_id]['MT'][peptideID]: ", ResponseInfo[patient_id]['MT'][peptideID], flush=True)

                    elif is_wt:
                        patientCOL = [patient_listing_numbers[i] for i in range(len(patient_listing_numbers)) if patient_listing_numbers[i] == patient_id][0]
                        print("Cansu patientCOL (wt seq): ",patientCOL, flush=True)
                        pID = int(peptideID.replace('WT',''))
                        resp = list(cd8_responses.loc[cd8_responses[pepID_EXCEL_COL] == pID][patientCOL])[0]
                        if math.isnan( resp ):
                            # del ResponseInfo[patient_id]['MT'][peptideID]
                            continue
                        else:
                            ResponseInfo[patient_id]['WT'][peptideID] = resp
                        print("Current seq, ResponseInfo[patient_id]['WT'][peptideID]: ", ResponseInfo[patient_id]['WT'][peptideID], flush=True)


                    # # Construct PatientID_PepID_Dict[PatientID] = [PeptideID]
                    if peptideID not in PatientID_PepID_Dict[patient_id]:
                        PatientID_PepID_Dict[patient_id].append(peptideID )
            else:
                print("else case. peptideID already in ResponseInfo.", flush=True)
    print("Returning.", flush=True)
    return ResponseInfo, PatientID_PepID_Dict, PeptideID_to_Sequence_Dict

def make_seqq_hla_to_Kd_dict(hla_seqq_dict, tempdir, use_netMHC34=False, use_netMHC40=True):
    seqq_hla_to_Kd_dict = {}
    for hla in hla_seqq_dict:

        nine_mers = hla_seqq_dict[hla]
        fastafile = os.path.join(tempdir,hla+'.fasta')
        with open(fastafile,'w') as faf:
            for item in nine_mers:

                epitope = str(item)
                epiID = ''.join([str(ord(i)) for i in item])
                faf.write(">" + str(epiID) + "\n" + epitope + "\n")
        print("In make_seqq_hla_to_Kd_dict(). fastafile created")

        # # Pass fasta file to netMHC, then load data from netMHC's output file.
        # # For netMHC 4.0, hla format has no ':'
        #hla_no_colon = hla.replace(':','')
        hla_no_colon = hla
        if use_netMHC34:
            print("Don't use netMHC34. Quitting")
            sys.exit()
            # nr = NetMHCReader() # netMHC34
            # outfile = nr.run_netMHC34(fastafile, hlas=[hla_no_colon])
            # nr.read_netMHC34(outfile, initialize=True)
        elif use_netMHC40:
            nr = NetMHC40()       # netMHC40
            outfile, kwargs, n_peptides = nr.run_netMHC(fastafile, hlas=[hla_no_colon]) # netMHC40
            kwargs['set_attribute'] = True
            print("In make_seqq_hla_to_Kd_dict(). outfile is: ",outfile, flush=True)
            nr.read_netMHC(outfile, **kwargs) # netMHC40


        print("In minKd_on_Cansu_sequences(). netHC completed. Now to fill seqq_hla_to_Kd_dict[(peptide,allele)]")
        # # Build L_human
        for peptide in nr.peptide_HLA_Kd:
            for allele in nr.peptide_HLA_Kd[peptide]:

                seqq_hla_to_Kd_dict[(peptide,allele)] = nr.peptide_HLA_Kd[peptide][allele]

    return seqq_hla_to_Kd_dict

def make_seqq_hla_to_Kd_dict_randomizedHLAs(hla_seqq_dict, tempdir, HLA_reshuffle_number, use_netMHC34=False, use_netMHC40=True):

    # # Randomly shuffle the hlas
    seqq_hla_to_Kd_dict = {}
    all_hlas = list(hla_seqq_dict.keys())
    random.seed(314+HLA_reshuffle_number)
    all_hlas_shuffled = []
    for h in all_hlas:

        # # If the same newHLA is chosen to replace two or more original HLAs,
        # # the <shuffled_hla_to_hla_dict> will only contain one of them.
        # # Don't let this happen.
        continu = True
        while continu:
            if 'HLA-A' in h:
                newHLA = random.sample(HLAA, 1)[0]
            elif 'HLA-B' in h:
                newHLA = random.sample(HLAB, 1)[0]
            elif 'HLA-C' in h:
                newHLA = random.sample(HLAC, 1)[0]
            else:
                print("hla doesnt contain HLA-A, HLA-B, HLA-C. Quitting")
                quit()
            if newHLA not in all_hlas_shuffled:
                continu = False

        all_hlas_shuffled.append(newHLA)

    # decodeIndexes = [all_hlas.index(x) for x in all_hlas_shuffled]
    # encodeIndexes = [all_hlas_shuffled.index(x) for x in all_hlas]

    hla_shuffled_hla_dict = {all_hlas[i]:all_hlas_shuffled[i] for i in range(len(all_hlas_shuffled))}
    shuffled_hla_to_hla_dict = {all_hlas_shuffled[i]:all_hlas[i] for i in range(len(all_hlas_shuffled))}

    print("all_hlas: ", all_hlas)
    print("all_hlas_shuffled", all_hlas_shuffled)
    print("hla_shuffled_hla_dict: ", hla_shuffled_hla_dict, flush=True)
    print("shuffled_hla_to_hla_dict: ", shuffled_hla_to_hla_dict, flush=True)
    print("HLA_reshuffle_number: ", HLA_reshuffle_number)

    for hla_orig in hla_seqq_dict:
        hla = hla_shuffled_hla_dict[hla_orig]

        nine_mers = hla_seqq_dict[hla]
        fastafile = os.path.join(tempdir,hla+'.fasta')
        with open(fastafile,'w') as faf:
            for item in nine_mers:

                epitope = str(item)
                epiID = ''.join([str(ord(i)) for i in item])
                faf.write(">" + str(epiID) + "\n" + epitope + "\n")
        print("In make_seqq_hla_to_Kd_dict(). fastafile created")

        # # Pass fasta file to netMHC, then load data from netMHC's output file.
        # # For netMHC 4.0, hla format has no ':'
        #hla_no_colon = hla.replace(':','')
        hla_no_colon = hla
        if use_netMHC34:
            print("Don't use netMHC34. Quitting")
            sys.exit()
            # nr = NetMHCReader() # netMHC34
            # outfile = nr.run_netMHC34(fastafile, hlas=[hla_no_colon])
            # nr.read_netMHC34(outfile, initialize=True)
        elif use_netMHC40:
            nr = NetMHC40()       # netMHC40
            outfile, kwargs, n_peptides = nr.run_netMHC(fastafile, hlas=[hla_no_colon]) # netMHC40
            kwargs['set_attribute'] = True
            print("In make_seqq_hla_to_Kd_dict(). outfile is: ",outfile, flush=True)
            nr.read_netMHC(outfile, **kwargs) # netMHC40


        print("In minKd_on_Cansu_sequences(). netHC completed. Now to fill seqq_hla_to_Kd_dict[(peptide,allele)]")
        # # Build L_human
        for peptide in nr.peptide_HLA_Kd:
            for allele in nr.peptide_HLA_Kd[peptide]:

                original_allele = shuffled_hla_to_hla_dict[allele]
                seqq_hla_to_Kd_dict[(peptide,original_allele)] = nr.peptide_HLA_Kd[peptide][allele]

    return seqq_hla_to_Kd_dict

def make_plot(c_x, c_y, xlab="X axis", ylab="Y axis", title="", ofile="", labels=None, fig=None, ax=None):

    print("In make_plot().", flush=True)
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(right=0.95, left=0.15, top=.80)

    c_pearson, c_pval = pearsonr(c_x, c_y)
    c_spearman, c_spval = spearmanr(c_x, c_y)
    c_kendall, c_kpval = kendalltau(c_x, c_y)
    print("Correlation stuff done. c_kendall=",c_kendall, flush=True)
    if labels is None:
        ax.plot(c_x, c_y, 'k.')
    else:
        ulabs = set(labels)
        for ulab in ulabs:
            dat = [x for x in zip(c_x, c_y, labels) if x[2] == ulab]
            c_x1 = [x[0] for x in dat]
            c_y1 = [x[1] for x in dat]
            ax.plot(c_x1, c_y1, '.')

    #ax.legend(fontsize=10, frameon=False, loc=2, bbox_to_anchor=(0, 1.20))
    #ax.legend(fontsize=10,loc='upper right') #MT
    title=title+'\nPearson coeff: %.2f (pval = %.2e),\nKendall coeff: %.2f (pval = %.2e)' % (
    c_pearson, c_pval, c_kendall, c_kpval)

    # plt.ylim(y_ax_arr)
    # plt.xlim(x_ax_arr)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)
    if ofile != "":
        plt.savefig(ofile)
    plt.close()

    print("In make_plot(). done.")

def make_plot_with_labels(X,Y,patient_labels=[],PoolLabels=[], wt_or_mt_label=[], fig_file="", title="", xlab="", ylab=""):
    c_pearson, c_pval = pearsonr(X, Y)
    c_spearman, c_spval = spearmanr(X, Y)
    c_kendall, c_kpval = kendalltau(X, Y)
    title=title+'\nPearson coeff: %.2f (pval = %.2e),\nKendall coeff: %.2f (pval = %.2e)' % (
    c_pearson, c_pval, c_kendall, c_kpval)

    # # we need to replace patient_labels entries with numbers in [0,len(set(patient_labels))-1]
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 1, 1)

    label_dict = {}
    for i, patID in enumerate( list(set(patient_labels))):
        label_dict[patID] = i
    color_labels = []
    for patID in patient_labels:
        color_labels.append( label_dict[patID])
    # # EACH PATIENT PLOTTED AS DISTINCT COLOR (multiple (hla, MT-WT) points
    # # per patient) See: https://stackoverflow.com/questions/12487060/matplotlib-color-according-to-class-labels

    N = len(set(patient_labels))
    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize
    bounds = np.linspace(0,N,N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)



    if PoolLabels == []:
        scat = ax1.scatter(X,Y, c=color_labels, cmap=cmap, norm=norm, alpha=0.4, marker ="o", s=5)

    else:
        X_alpha = [X[i] for i in range(len(X)) if PoolLabels[i]==0]
        Y_alpha = [Y[i] for i in range(len(Y)) if PoolLabels[i]==0]
        color_labels_alpha = [color_labels[i] for i in range(len(color_labels)) if PoolLabels[i]==0]
        scat = ax1.scatter(X_alpha,Y_alpha, c=color_labels_alpha, cmap=cmap, norm=norm, alpha=0.9, marker ="v", s=26)

        # create the colorbar
        cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
        cb.set_label('Patient Index')

        X_beta = [X[i] for i in range(len(X)) if PoolLabels[i]==1]
        Y_beta = [Y[i] for i in range(len(Y)) if PoolLabels[i]==1]
        color_labels_beta = [color_labels[i] for i in range(len(color_labels)) if PoolLabels[i]==1]
        scat = ax1.scatter(X_beta,Y_beta, c=color_labels_beta, cmap=cmap, norm=norm, alpha=0.6, marker ="s", s=26)

        X_gamma = [X[i] for i in range(len(X)) if PoolLabels[i]==2]
        Y_gamma = [Y[i] for i in range(len(Y)) if PoolLabels[i]==2]
        color_labels_gamma = [color_labels[i] for i in range(len(color_labels)) if PoolLabels[i]==2]
        scat = ax1.scatter(X_gamma,Y_gamma, c=color_labels_gamma, cmap=cmap, norm=norm, alpha=0.8, marker ="D", s=26)

        leg = plt.legend(["alpha", "beta", "gamma"])
        ax1.add_artist(leg)


    if wt_or_mt_label:
        # # Add red marker to points that are MUTMUT
        X_mut = [X[i] for i in range(len(X)) if wt_or_mt_label[i] == 2]
        Y_mut = [Y[i] for i in range(len(Y)) if wt_or_mt_label[i] == 2]
        scat = ax1.scatter(X_mut, Y_mut, c='r', alpha=1.0, marker ="o", s=6)
        scat = ax1.scatter(X_mut, Y_mut, c='b', alpha=0.2, marker ="o", s=6) # make the red purplueish

        # # Add black marker to points that are WTMUT
        X_mut = [X[i] for i in range(len(X)) if wt_or_mt_label[i] == 1]
        Y_mut = [Y[i] for i in range(len(Y)) if wt_or_mt_label[i] == 1]
        scat = ax1.scatter(X_mut, Y_mut, c='k', alpha=1.0, marker ="o", s=6)

        # # Add cyan marker to points that are MUTWT
        X_mut = [X[i] for i in range(len(X)) if wt_or_mt_label[i] == 3]
        Y_mut = [Y[i] for i in range(len(Y)) if wt_or_mt_label[i] == 3]
        scat = ax1.scatter(X_mut, Y_mut, c='cyan', alpha=1.0, marker ="o", s=6)

    ax1.set_title(title)
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab)
    # ax1.set_xscale('log')
    plt.savefig(fig_file, dpi=600)

def aggr_append(X, Q, aggrFun_ninemers, a_soft=0):

    # # Aggregate over wt 9mers
    if aggrFun_ninemers == 'max':
        X.append( max(Q) )
    elif aggrFun_ninemers == 'min':
        X.append( min(Q) )
    elif aggrFun_ninemers == 'sum':
        X.append( sum(Q) )
    elif aggrFun_ninemers == 'gmean':
        X.append( gmean(Q) )
    elif aggrFun_ninemers == 'mean':
        X.append( sum(Q)/len(Q) )
    elif aggrFun_ninemers == 'LSE':
        X.append( LSE(Q) )
    elif aggrFun_ninemers == 'realSoftMax':
        r = realSoftMax(Q,a_soft)
        if math.isnan(r):
            print("Q, a_soft: ", Q, a_soft)
            print("Quitting")
            sys.exit()
        X.append( r )
    elif aggrFun_ninemers == 'realSoftMin':
        r = realSoftMin(Q,a_soft)
        X.append( r )

    return X
def presentation(seqq, hla, seqq_hla_to_Kd_dict, conc, denom):

    Kd = seqq_hla_to_Kd_dict[(seqq,hla)]

    if use_conc:
        numerator = conc/Kd
        p = numerator/denom
    else:
        p = 1/Kd

    return p

def compute_pool_peptide_ninemers(poolMutations, long_wt_mt_pair__to__mut_dict, mut_to_wt15merList_dict, mut_to_mt15merList_dict):
    '''
    returns
    w_pool_w_peptide_ninemers = [[long wt peptide, list of [wt,wt] ninemer pairs]]
    w_pool_w_peptide_ninemers = [[long mt peptide, list of [mt,mt] ninemer pairs]]

    '''
    w_pool_w_peptide_ninemers = []
    m_pool_m_peptide_ninemers = []

    wt_mt_mutations = [ [key,long_wt_mt_pair__to__mut_dict[key]] for key in
        long_wt_mt_pair__to__mut_dict if long_wt_mt_pair__to__mut_dict[key] in poolMutations]

    # # Iterate over mutations
    for item in wt_mt_mutations:

        long_wt_sequence = item[0][0]
        long_mt_sequence = item[0][1]
        mutation = item[1]

        ww_peptide_ninemers = []
        mm_peptide_ninemers = []

        # # -------------------------------------------------------------------
        # # Construct wm_pool_ninemers - pairs of ninemers for
        # # each (wt_15mer,mt_15mer).
        for e, wt_15mer in enumerate(mut_to_wt15merList_dict[mutation]):
            mt_15mer = mut_to_mt15merList_dict[mutation][e]

            wt_pool_ninemers2 = generate_9_mers(wt_15mer)
            mt_pool_ninemers2 = generate_9_mers(mt_15mer)

            # # Iterate over 9mer pairs for current (WT long peptide, MT long peptide)
            for j in range( len( wt_pool_ninemers2)):
                wt_seqq = wt_pool_ninemers2[j]
                mt_seqq = mt_pool_ninemers2[j]


                # # Keep unique wtwt pairs
                if [wt_seqq,wt_seqq] not in ww_peptide_ninemers:
                    ww_peptide_ninemers.append([wt_seqq,wt_seqq])

                # # Keep unique mtmt pairs
                if [mt_seqq,mt_seqq] not in mm_peptide_ninemers:
                    mm_peptide_ninemers.append([mt_seqq,mt_seqq])

        # # -------------------------------------------------------------------
        w_pool_w_peptide_ninemers.append( [long_wt_sequence, ww_peptide_ninemers])
        m_pool_m_peptide_ninemers.append( [long_mt_sequence, mm_peptide_ninemers])

    return w_pool_w_peptide_ninemers, m_pool_m_peptide_ninemers

def compute_pool_ninemers(poolMutations, long_wt_mt_pair__to__mut_dict,
                          mut_to_wt15merList_dict, mut_to_mt15merList_dict):

    # wt_pool_ninemers = []
    # mt_pool_ninemers = []
    all_ninemers_set = set()

    wm_pool_ninemers = []
    ww_pool_ninemers = []
    mm_pool_ninemers = []
    mw_pool_ninemers = []


    wt_mt_mutations = [ long_wt_mt_pair__to__mut_dict[key] for key in
        long_wt_mt_pair__to__mut_dict if long_wt_mt_pair__to__mut_dict[key] in poolMutations]

    # # Iterate over mutations
    for mutation in wt_mt_mutations:

        # # # Construct wt_pool_ninemers, mt_pool_ninemers.
        # # # Iterate over 15mers within the long peptides corresponding to
        # # # this mutation.
        # for wt_15mer in mut_to_wt15merList_dict[mutation]:
        #     wt_pool_ninemers.extend( generate_9_mers(wt_15mer) )
        # for mt_15mer in mut_to_mt15merList_dict[mutation]:
        #     mt_pool_ninemers.extend( generate_9_mers(mt_15mer) )

        # # -------------------------------------------------------------------
        # # Construct wm_pool_ninemers - pairs of ninemers for
        # # each (wt_15mer,mt_15mer).
        for e, wt_15mer in enumerate(mut_to_wt15merList_dict[mutation]):
            mt_15mer = mut_to_mt15merList_dict[mutation][e]

            wt_pool_ninemers2 = generate_9_mers(wt_15mer)
            mt_pool_ninemers2 = generate_9_mers(mt_15mer)

            # # Iterate over 9mer pairs for current (WT long peptide, MT long peptide)
            for j in range( len( wt_pool_ninemers2)):
                wt_seqq = wt_pool_ninemers2[j]
                mt_seqq = mt_pool_ninemers2[j]
                vaccine_seqq = wt_seqq
                # # Keep unique 9mer pairs over sliding window
                if [wt_seqq,mt_seqq] not in wm_pool_ninemers:
                    wm_pool_ninemers.append([wt_seqq,mt_seqq,vaccine_seqq])

                # # Keep unique wtwt pairs
                if [wt_seqq,wt_seqq] not in ww_pool_ninemers:
                    ww_pool_ninemers.append([wt_seqq,wt_seqq,vaccine_seqq])

                # # Keep unique mtmt pairs
                if [mt_seqq,mt_seqq] not in mm_pool_ninemers:
                    mm_pool_ninemers.append([mt_seqq,mt_seqq,vaccine_seqq])

                # # Keep unique mtwt pairs
                if [mt_seqq,wt_seqq] not in mw_pool_ninemers:
                    mw_pool_ninemers.append([mt_seqq,wt_seqq,vaccine_seqq])

                all_ninemers_set.add(wt_seqq)
                all_ninemers_set.add(mt_seqq)
        # # -------------------------------------------------------------------


    return wm_pool_ninemers, ww_pool_ninemers, mm_pool_ninemers, mw_pool_ninemers, all_ninemers_set

def compute_min_d_IEDB_dict(min_distance_dict, ninemers):
    # min_distance_dict = {}

    if model['IEDB'] == 'Everything':
        pepd = PeptideDistance(enemydb="2023_05",nlen=9)
        for ninemer in ninemers:
            min_distance_dict[ninemer] = pepd.closest(ninemer)[0]
    elif model['IEDB'] == 'No_Covid':
        pepd = PeptideDistance(enemydb="2023_05_nocovid",nlen=9)
        for ninemer in ninemers:
            min_distance_dict[ninemer] = pepd.closest(ninemer)[0]
    elif model['IEDB'] == 'Only_Covid':
        pepd = PeptideDistance(enemydb="2023_05_covid",nlen=9)
        for ninemer in ninemers:
            min_distance_dict[ninemer] = pepd.closest(ninemer)[0]

    return min_distance_dict

def HFun(d):
    if d <= 1:
        return 0
    else:
        return math.inf

def eval_ninemer_func(model, X_hlas, hla, BetaParam, BetaParam_IEDB, a_soft,
                      aggrFun_ninemers, pool_ninemers,seqq_hla_to_Kd_dict,
                      conc, conc_vaccine, ninemer_pairwise_dist_dict):

    '''
    The vaccination component is always WT. The vaccine 9mers will be the index-2
    element of each 3-tuple in pool_ninemers.

    '''
    # # Let this method know T_dict is a global variable. This allows us to
    # # modigy the global variable (rather than just access it)
    # # preT_dict does not need to be declared as global here because we are not
    # # modifying it, only accessing it, and it is already a global variable.
    global T_dict
    global EpiHLA_to_H_to_SelfEpis
    global EpiHLA_to_H_to_SelfSethnaDists
    global EpiHLA_to_H_to_SelfKdsInv


    Q_9mer = []

    if use_conc:
        denom_stimPool = sum( [conc/seqq_hla_to_Kd_dict[(pool_ninemers[s][0],hla)] for s in range(len(pool_ninemers))] ) + 1
        denom_restimPool = sum( [conc/seqq_hla_to_Kd_dict[(pool_ninemers[s][1],hla)] for s in range(len(pool_ninemers))] ) + 1
    else:
        denom_stimPool = 0
        denom_restimPool = 0

    if model['include_vaccine']:
        wuhan_spike_ninemers = generate_9_mers(WUHAN_Spike)
        denom_vaccine = sum( [conc_vaccine/seqq_hla_to_Kd_dict[(wuhan_spike_ninemers[s],hla)] for s in range(len(wuhan_spike_ninemers))] ) + 1

    for item in pool_ninemers:

        seqq_stimPool = item[0]
        seqq_restimPool = item[1]

        pres_seqq_restimPool = presentation(seqq_restimPool, hla, seqq_hla_to_Kd_dict, conc, denom_restimPool)

        if include_first_stimulation:
            pres_seqq_stimPool = presentation(seqq_stimPool, hla, seqq_hla_to_Kd_dict, conc, denom_stimPool)
            d_ab = ninemer_pairwise_dist_dict[seqq_stimPool][seqq_restimPool]
        else:
            pres_seqq_stimPool = 1.0
            d_ab = 0.0

        if model['use_tolerance']:

            # # UPDATE global variable <T_dict> if necessary
            if (seqq_stimPool,hla) not in T_dict:

                # # Compute dicts containing preT_dict info, but partitioned by
                # # Hamming distance, then compute Tolerance dict entry.
                if (seqq_stimPool,hla) not in EpiHLA_to_H_to_SelfEpis:
                    tpee = time.time()
                    E_SelfEpis, E_SelfSethnaDists, E_SelfKdsInv = create_self_dicts((seqq_stimPool,hla), preT_dict[(seqq_stimPool,hla)])
                    EpiHLA_to_H_to_SelfEpis.update(E_SelfEpis)
                    EpiHLA_to_H_to_SelfSethnaDists.update(E_SelfSethnaDists)
                    EpiHLA_to_H_to_SelfKdsInv.update(E_SelfKdsInv)
                ds = []
                Kds_inv = []
                for hammingDist in range(hamming_cutoff+1):
                    ds.extend( EpiHLA_to_H_to_SelfSethnaDists[(seqq_stimPool,hla)][hammingDist] )
                    Kds_inv.extend( EpiHLA_to_H_to_SelfKdsInv[(seqq_stimPool,hla)][hammingDist] )
                if len(ds) == 0:
                    continue

                # log_KsInverse = Utils.log_sum(-gammat*np.array(Kds_inv) + np.log(Kds_inv))
                # log_KsInverse = Utils.log_sum(-gammat*np.array(ds) + np.log(Kds_inv))

                # ds = np.array(ds)*np.array(Kds_inv)
                # log_KsInverse = Utils.log_sum(-gammat*ds + np.log(Kds_inv))

                ds = np.array(Kds_inv)/np.array(ds)
                log_KsInverse = Utils.log_sum(-gammat*ds + np.log(Kds_inv))
                T_dict[(seqq_stimPool,hla)] = 1.0 /( 1 + K0*np.exp(log_KsInverse) )

            if (seqq_restimPool,hla) not in T_dict:

                # # Compute dicts containing preT_dict info, but partitioned by
                # # Hamming distance, then compute Tolerance dict entry.
                if (seqq_restimPool,hla) not in EpiHLA_to_H_to_SelfEpis:
                    tpee = time.time()
                    E_SelfEpis, E_SelfSethnaDists, E_SelfKdsInv = create_self_dicts((seqq_restimPool,hla), preT_dict[(seqq_restimPool,hla)])
                    EpiHLA_to_H_to_SelfEpis.update(E_SelfEpis)
                    EpiHLA_to_H_to_SelfSethnaDists.update(E_SelfSethnaDists)
                    EpiHLA_to_H_to_SelfKdsInv.update(E_SelfKdsInv)
                ds = []
                Kds_inv = []

                # # Use Boltzman version.
                # # We want the distances and Kds_inv for the
                # # current (9mer,assay_hla)
                for hammingDist in range(hamming_cutoff+1):
                    ds.extend( EpiHLA_to_H_to_SelfSethnaDists[(seqq_restimPool,hla)][hammingDist] )
                    Kds_inv.extend( EpiHLA_to_H_to_SelfKdsInv[(seqq_restimPool,hla)][hammingDist] )
                if len(ds) == 0:
                    continue

                # log_KsInverse = Utils.log_sum(-gammat*np.array(Kds_inv) + np.log(Kds_inv))
                # log_KsInverse = Utils.log_sum(-gammat*np.array(ds) + np.log(Kds_inv))

                # ds = np.array(ds)*np.array(Kds_inv)
                # log_KsInverse = Utils.log_sum(-gammat*np.array(ds) + np.log(Kds_inv))

                ds = np.array(Kds_inv)/np.array(ds)
                log_KsInverse = Utils.log_sum(-gammat*np.array(ds) + np.log(Kds_inv))
                T_dict[(seqq_restimPool,hla)] = 1.0 /( 1 + K0*np.exp(log_KsInverse) )

            # # USE Tolerance values
            T_seqq_stimPool = T_dict[(seqq_stimPool,hla)]
            T_seqq_restimPool = T_dict[(seqq_restimPool,hla)]
            pres_seqq_stimPool *= T_seqq_stimPool
            pres_seqq_restimPool *= T_seqq_restimPool

        if model['include_vaccine']:
            seqq_vaccine = item[2]
            pres_seqq_vaccine = presentation(seqq_vaccine, hla, seqq_hla_to_Kd_dict, conc_vaccine, denom_vaccine)
            d_va = ninemer_pairwise_dist_dict[seqq_vaccine][seqq_stimPool]
            d_vb = ninemer_pairwise_dist_dict[seqq_vaccine][seqq_restimPool]
            F = pres_seqq_vaccine * pres_seqq_stimPool * pres_seqq_restimPool * np.exp(-BetaParam*(d_ab + d_va + d_vb))

        else:
            if model['IEDB'] == 'No_IEDB':
                F = pres_seqq_stimPool * pres_seqq_restimPool * np.exp(-BetaParam*(d_ab))
            else:
                #print("min_d_IEDB_dict.keys(): ",list(min_d_IEDB_dict.keys()))
                min_d_IEDB = min_d_IEDB_dict[seqq_restimPool]
                F = pres_seqq_stimPool * pres_seqq_restimPool * np.exp( -BetaParam*(d_ab) -BetaParam_IEDB*min_d_IEDB )

        Q_9mer.append( F )

    # # Aggregate over 9mers
    X_hlas = aggr_append(X_hlas, Q_9mer, aggrFun_ninemers, a_soft=a_soft)
    return X_hlas, Q_9mer

def compute_XY_poolStim_peptideRestim(aggrFun_ninemers, a_soft, poolDict,
               ResponseInfo, poolMutations,
               patientID_to_seq_hla_dict, seqq_hla_to_Kd_dict,
               long_wt_mt_pair__to__mut_dict, BetaParam, BetaParam_IEDB,
               PeptideID_to_Sequence_Dict, ninemer_pairwise_dist_dict,
               ninemer_pairwise_dist_dict_hamming,
               w_pool_w_peptide_ninemers, m_pool_m_peptide_ninemers,
               conc, conc_vaccine,
               include_WTWT=True, include_MUTMUT=True):
    '''
    include_PoolStim_PeptideRestim: This is the original data set from Tim,
    Vaccinated cohort summary of mutations deconvolutions for predictions updated 20220228-SPIKEpeptides.numbers
    Post vaccination, stimulate with WT (/MT) pool,
                        restimulate with WT (/MT) *peptide* (not with entire pool)

    1.
    Since restimulation now occurs for individual peptides rather than
    for the entire pool, we now have <#peptides> experimental measurements
    for every single-pool measurement in the poolStim_poolRestim (i.e., WTWT,
                                                WTMUT, MUTMUT) context.

    2. THE IMPLICATION IS THAT...
    [ aggrFun_ninemers: is function for aggregating 9mers derived from long
                      peptides in the current pool (i.e., alpha, beta, gamma),
                      ***within a single WT (/MT) long peptide***. ]

    WHEREAS... In the poolStim_poolRestim context, ninemer aggregation
    was over ALL 9mers derived from long peptides in the
    pool.

    3. We also need to reverse the order of hla loop and seq loop. I.e., the
       hla aggregation happens for *each* long peptide separately, so the hla
       loop is now the inner loop.



    '''
    # # -----------------------------------------------------------------------

    Sequence_to_PeptideID_Dict = {}
    for pepID in PeptideID_to_Sequence_Dict:
        seq = PeptideID_to_Sequence_Dict[pepID]
        Sequence_to_PeptideID_Dict[seq] = pepID
    # # -----------------------------------------------------------------------
    # # E.g., Q_dict[patient_id][hla]['WTWT'] = ninemer_qualities_preAggr
    Q_dict = {}
    Y_list = []
    X_list = []
    WW_vs_WM_vs_MM_vs_MW_list = [] # keep track of whether X came from  'WT' (0) or 'MT' (1)
    patient_id_list = []
    for patient_id in poolDict['WTWT']:

        if patient_id not in Q_dict:
            Q_dict[patient_id] = {}

        # # patientID_to_seq_hla_dict[patient_id] = [[seq,hla]]
        patient_hlas = [patientID_to_seq_hla_dict[patient_id][i][1] for i in range(len(patientID_to_seq_hla_dict[patient_id])) ]
        patient_hlas = list( set(patient_hlas))

        # # Iterate over restimulation peptides
        for i in range(len(w_pool_w_peptide_ninemers)):
            # # Get long peptide
            wt_seq = w_pool_w_peptide_ninemers[i][0] #wt_seqs[i]
            mt_seq = m_pool_m_peptide_ninemers[i][0] #mt_seqs[i]

            # # Get pairs of 9mers for current long restimulation peptide
            ww_ninemers = w_pool_w_peptide_ninemers[i][1]
            mm_ninemers = m_pool_m_peptide_ninemers[i][1]


            # # Add the vaccination (WT) seqq to each tuple in ww_ninemers/mm_ninemers
            for j,item in enumerate(ww_ninemers):
                ww_ninemers[j].append(item[0])
                mm_ninemers[j].append((item[0]))



            X_ww_hlas = [] # Model values across 9mers in the current pool, in the current WT sequence
            X_mm_hlas = [] # Model values across 9mers in the current pool, in the current MT sequence

            for hla in patient_hlas:

                if hla not in Q_dict[patient_id]:
                    Q_dict[patient_id][hla] = {}

                if include_WTWT:
                    X_ww_hlas, ninemer_qualities_preAggr = eval_ninemer_func(model, X_ww_hlas, hla,
                            BetaParam, BetaParam_IEDB, a_soft, aggrFun_ninemers,
                            ww_ninemers, seqq_hla_to_Kd_dict,
                            conc, conc_vaccine, ninemer_pairwise_dist_dict)#, T_dict=T_dict)
                    Q_dict[patient_id][hla]['WTWT'] = ninemer_qualities_preAggr

                if include_MUTMUT:
                    X_mm_hlas, ninemer_qualities_preAggr = eval_ninemer_func(model, X_mm_hlas, hla,
                            BetaParam, BetaParam_IEDB, a_soft, aggrFun_ninemers,
                            mm_ninemers, seqq_hla_to_Kd_dict,
                            conc, conc_vaccine, ninemer_pairwise_dist_dict)#, T_dict=T_dict)
                    Q_dict[patient_id][hla]['MUTMUT'] = ninemer_qualities_preAggr



            # # Aggregate (sum) over hlas, append to model data.
            # # Append order (WTWT, WTMUT, MTMT)  must be same for
            # # X_list, Y_list, WW_vs_WM_vs_MM_vs_MW_list.

            # # Since we are retrieving experimental data from ResponseInfo, we
            # # need to handle the fact that missing data was excluded from
            # # ResponseInfo during its earlier construction.
            if Sequence_to_PeptideID_Dict[wt_seq] in ResponseInfo[patient_id]['WT']:
                if include_WTWT:
                    X_list.append( sum( X_ww_hlas) )
                    Y_list.append(ResponseInfo[patient_id]['WT'][Sequence_to_PeptideID_Dict[wt_seq]])
                    WW_vs_WM_vs_MM_vs_MW_list.append(0) #WTWT
                    patient_id_list.append(patient_id)
            if Sequence_to_PeptideID_Dict[mt_seq] in ResponseInfo[patient_id]['MT']:
                if include_MUTMUT:
                    X_list.append( sum( X_mm_hlas) )
                    Y_list.append(ResponseInfo[patient_id]['MT'][Sequence_to_PeptideID_Dict[mt_seq]])
                    WW_vs_WM_vs_MM_vs_MW_list.append(2) #MUTMUT
                    patient_id_list.append(patient_id)

    return Q_dict, X_list, Y_list, patient_id_list, WW_vs_WM_vs_MM_vs_MW_list


def compute_XY_poolStim_poolRestim(aggrFun_ninemers, a_soft, poolDict, poolMutations,
               patientID_to_seq_hla_dict, seqq_hla_to_Kd_dict,
               long_wt_mt_pair__to__mut_dict, BetaParam, BetaParam_IEDB, ninemer_pairwise_dist_dict,
               ninemer_pairwise_dist_dict_hamming,
               ww_pool_ninemers, mm_pool_ninemers , wm_pool_ninemers, mw_pool_ninemers,
               conc, conc_vaccine,
               include_WTWT=True, include_WTMUT=True,
               include_MUTMUT=True, include_MUTWT=False):
    '''
    PoolStim_PoolRestim: This is the pool level dataset from Cansu,
    Vaccinated cohort summary of mutation deconvolutions for predictions_updated.xlsx
    Post vaccination, stimulate with WT (/MT) pool,
                        restimulate with WT (/MT) pool

    aggrFun_ninemers: function for aggregating 9mers derived from long
                        peptides in the current pool (i.e., alpha, beta, gamma)

    include_WTWT: WTWT means post vaccination, stimulate with WT members of
                    the current pool. After 10 days of putative Tcell expansion
                    (by which time IFG levels have decreased)
                    restimulate with WT. The measurement is then the percent
                    of Tcells expressing IFG.
    include_WTMUT, include_MUTMUT, include_MUTWT are analogous.

    '''
    # # E.g., Q_dict[patient_id][hla]['WTWT'] = ninemer_qualities_preAggr
    Q_dict = {}
    Y_list = []
    X_list = []
    WW_vs_WM_vs_MM_vs_MW_list = [] # keep track of whether X came from  'WT' (0) or 'MT' (1)
    patient_id_list = []

    for patient_id in poolDict['WTWT']:

        if patient_id not in Q_dict:
            Q_dict[patient_id] = {}

        # # patientID_to_seq_hla_dict[patient_id] = [[seq,hla]]
        patient_hlas = [patientID_to_seq_hla_dict[patient_id][i][1] for i in range(len(patientID_to_seq_hla_dict[patient_id])) ]
        patient_hlas = list( set(patient_hlas))


        X_ww_hlas = [] # Model values across 9mers in the WT set (in the current pool)
        X_wm_hlas = [] # Model values across 9mers in the WT set and MT set (in the current pool)
        X_mw_hlas = []
        X_mm_hlas = [] # Model values across 9mers in the MT set (in the current pool)


        # # EACH PATIENT SEES ALL 9MERS PRESENT IN THE POOL. THE HLA LOOP
        # # THEREFORE BE THE OUTER LOOP. THE LOOP OVER LONG PEPTIDES (AND
        # # NINEMERS) SHOULD BE INNER LOOPS.
        for hla in patient_hlas:

            if hla not in Q_dict[patient_id]:
                Q_dict[patient_id][hla] = {}

            if include_WTWT:
                X_ww_hlas, ninemer_qualities_preAggr = eval_ninemer_func(model, X_ww_hlas, hla,
                        BetaParam, BetaParam_IEDB, a_soft, aggrFun_ninemers,
                        ww_pool_ninemers, seqq_hla_to_Kd_dict,
                        conc, conc_vaccine, ninemer_pairwise_dist_dict)
                Q_dict[patient_id][hla]['WTWT'] = ninemer_qualities_preAggr

            if include_WTMUT:
                X_wm_hlas, ninemer_qualities_preAggr = eval_ninemer_func(model, X_wm_hlas, hla,
                        BetaParam, BetaParam_IEDB, a_soft, aggrFun_ninemers,
                        wm_pool_ninemers, seqq_hla_to_Kd_dict,
                        conc, conc_vaccine, ninemer_pairwise_dist_dict)
                Q_dict[patient_id][hla]['WTMUT'] = ninemer_qualities_preAggr
            if include_MUTWT:
                X_mw_hlas, ninemer_qualities_preAggr = eval_ninemer_func(model, X_mw_hlas, hla,
                        BetaParam, BetaParam_IEDB, a_soft, aggrFun_ninemers,
                        mw_pool_ninemers, seqq_hla_to_Kd_dict,
                        conc, conc_vaccine, ninemer_pairwise_dist_dict)
                Q_dict[patient_id][hla]['MUTWT'] = ninemer_qualities_preAggr
            if include_MUTMUT:
                X_mm_hlas, ninemer_qualities_preAggr = eval_ninemer_func(model, X_mm_hlas, hla,
                        BetaParam, BetaParam_IEDB, a_soft, aggrFun_ninemers,
                        mm_pool_ninemers, seqq_hla_to_Kd_dict,
                        conc, conc_vaccine, ninemer_pairwise_dist_dict)
                Q_dict[patient_id][hla]['MUTMUT'] = ninemer_qualities_preAggr

        # print("BetaParam_IEDB, patient_hlas = ", BetaParam_IEDB, patient_hlas)
        # print("X_ww_hlas = ", X_ww_hlas)
        # print("X_wm_hlas = ", X_wm_hlas)
        # print("X_mw_hlas = ", X_mw_hlas)
        # print("X_mm_hlas = ", X_mm_hlas)

        # # Aggregate (sum) over hlas, append to model data.
        # # Append order (WTWT, WTMUT, MTMT)  must be same for
        # # X_list, Y_list, WW_vs_WM_vs_MM_vs_MW_list.
        if include_WTWT:
            X_list.append( sum( X_ww_hlas) )
            Y_list.append(poolDict['WTWT'][patient_id])
            WW_vs_WM_vs_MM_vs_MW_list.append(0) #WTWT
            patient_id_list.append(patient_id)

        if include_WTMUT:
            X_list.append( sum( X_wm_hlas) )
            Y_list.append(poolDict['WTMUT'][patient_id])
            WW_vs_WM_vs_MM_vs_MW_list.append(1) #WTMUT
            patient_id_list.append(patient_id)

        if include_MUTMUT:
            X_list.append( sum( X_mm_hlas) )
            Y_list.append(poolDict['MUTMUT'][patient_id])
            WW_vs_WM_vs_MM_vs_MW_list.append(2) #MUTMUT
            patient_id_list.append(patient_id)

        if include_MUTWT:
            X_list.append( sum( X_mw_hlas) )
            Y_list.append(poolDict['MUTWT'][patient_id])
            WW_vs_WM_vs_MM_vs_MW_list.append(3) #MUTMUT
            patient_id_list.append(patient_id)

    return Q_dict, X_list, Y_list, patient_id_list, WW_vs_WM_vs_MM_vs_MW_list


def applyNormalization_intraPool(X_allPools, Y_allPools, Patients_allPools, PoolLabels, applyX=True, applyY=True,
                                 random_permute=False):
    # # Additively normalize X,Y
    '''
    Normalize within patient, within pool.
    x_i <- x_i - ave(x_1,...x_6)
    y_i <- y_i - ave(y_1,...y_6)
    '''

    '''
    Marta isn't sure we should normalize within pools for 2 reasons:
        1. there are only 3 measurements to normalize over - outliers a problem
        2. the pools have already been experimentally normalized by Cansu
        3. without a reason to normalize within pool, doing so could
            raise the correlation in a non-meaningful way.
    My perspective is:
        1. normalizing within pools leads to across the board higher correlations
        2. if it's not harmful, why not

    The question is, is normalizing within pools introducing some unwanted bias
    that artificially inflates correlation values?
    To do so, try normalizing within 'randomly' defined pools instead.
    If her point 3 is valid, even these artifical pools should artificially
    inflate the correlation results.

    '''
    if random_permute:
        #PoolLabels = list(np.random.permutation(PoolLabels))
        PoolLabels = [1,
         1,
         1,
         0,
         0,
         1,
         0,
         2,
         2,
         2,
         1,
         1,
         0,
         2,
         0,
         2,
         0,
         0,
         0,
         2,
         1,
         1,
         2,
         1,
         2,
         1,
         2,
         0,
         0,
         0,
         0,
         0,
         1,
         0,
         1,
         0,
         2,
         1,
         1,
         0,
         1,
         2,
         1,
         0,
         2,
         0,
         1,
         2,
         0,
         2,
         1,
         1,
         0,
         2,
         0,
         0,
         2,
         1,
         0,
         0,
         2,
         0,
         2,
         2,
         2,
         2,
         0,
         0,
         1,
         1,
         0,
         1,
         2,
         1,
         1,
         2,
         1,
         2,
         2,
         2,
         0,
         2,
         2,
         1,
         1,
         0,
         2,
         1,
         2,
         2,
         0,
         1,
         0,
         1,
         0,
         2,
         2,
         1,
         0,
         2,
         1,
         0,
         2,
         1,
         1,
         0,
         2]

    # # First determine lists needed for normalization.
    X_Patient_Dict = {} # patient_Dict[patient] = [x1,x2,...,x6]
    Y_Patient_Dict = {}
    for i,x in enumerate(X_allPools):
        patient = Patients_allPools[i]
        y = Y_allPools[i]
        pool_label = PoolLabels[i]

        if patient not in X_Patient_Dict:
            X_Patient_Dict[patient] = {}
            Y_Patient_Dict[patient] = {}

            X_Patient_Dict[patient][pool_label] = [x]
            Y_Patient_Dict[patient][pool_label] = [y]

        else:
            if pool_label in X_Patient_Dict[patient]:
                X_Patient_Dict[patient][pool_label].append(x)
                Y_Patient_Dict[patient][pool_label].append(y)
            else:
                X_Patient_Dict[patient][pool_label] = [x]
                Y_Patient_Dict[patient][pool_label] = [y]

    # # Now additively normalize
    # # " This way you will compare relative measurements only, and you will disregard patient-specific effects"
    for i,x in enumerate(X_allPools):
        patient = Patients_allPools[i]
        pool_label = PoolLabels[i]
        if applyX:
            if len(X_Patient_Dict[patient]) > 1:
                X_allPools[i] = X_allPools[i] - ( sum(X_Patient_Dict[patient][pool_label]) / len(X_Patient_Dict[patient][pool_label]) )
        if applyY:
            if len(Y_Patient_Dict[patient]) > 1:
                Y_allPools[i] = Y_allPools[i] - ( sum(Y_Patient_Dict[patient][pool_label]) / len(Y_Patient_Dict[patient][pool_label]) )

    # # Now normalize by std within patient, within pool_label
    for i,x in enumerate(X_allPools):
        patient = Patients_allPools[i]
        pool_label = PoolLabels[i]
        # # Y_allPools may contain 0s. Don't div by std in this case.
        if applyY:
            if Y_allPools[i] != 0 and len(Y_Patient_Dict[patient]) > 1:
                Y_allPools[i] = Y_allPools[i] / np.std(Y_Patient_Dict[patient][pool_label])
        if applyX:
            if X_allPools[i] != 0 and len(X_Patient_Dict[patient]) > 1:
                X_allPools[i] = X_allPools[i] / np.std(X_Patient_Dict[patient][pool_label])

    return X_allPools, Y_allPools

def applyNormalization(X_allPools,Y_allPools, Patients_allPools, applyX=True, applyY=True):
    # # Additively normalize X,Y
    '''
    Normalize within patient.
    x_i <- x_i - ave(x_1,...x_6)
    y_i <- y_i - ave(y_1,...y_6)
    '''
    # # First determine lists needed for normalization.
    X_Patient_Dict = {} # patient_Dict[patient] = [x1,x2,...,x6]
    Y_Patient_Dict = {}
    for i,x in enumerate(X_allPools):
        patient = Patients_allPools[i]
        y = Y_allPools[i]

        if patient not in X_Patient_Dict:
            X_Patient_Dict[patient] = [x]
            Y_Patient_Dict[patient] = [y]
        else:
            X_Patient_Dict[patient].append(x)
            Y_Patient_Dict[patient].append(y)

    # # Now additively normalize
    # # " This way you will compare relative measurements only, and you will disregard patient-specific effects"
    for i,x in enumerate(X_allPools):
        patient = Patients_allPools[i]
        if applyX:
            if len(X_Patient_Dict[patient]) > 1:
                X_allPools[i] = X_allPools[i] - ( sum(X_Patient_Dict[patient]) / len(X_Patient_Dict[patient]) )
        if applyY:
            if len(Y_Patient_Dict[patient]) > 1:
                Y_allPools[i] = Y_allPools[i] - ( sum(Y_Patient_Dict[patient]) / len(Y_Patient_Dict[patient]) )

    # # Now normalize by std within patient
    for i,x in enumerate(X_allPools):
        patient = Patients_allPools[i]
        # # Y_allPools may contain 0s. Don't div by std in this case.
        if applyY:
            if Y_allPools[i] != 0 and len(Y_Patient_Dict[patient]) > 1:
                Y_allPools[i] = Y_allPools[i] / np.std(Y_Patient_Dict[patient])
        if applyX:
            if X_allPools[i] != 0 and len(X_Patient_Dict[patient]) > 1:
                X_allPools[i] = X_allPools[i] / np.std(X_Patient_Dict[patient])

    return X_allPools, Y_allPools

def handleMissingData(X,Y,patient_id_list, W_or_M):
    # print("In handleMissingData(). X: ", X)
    # print("In handleMissingData(). Y: ", Y)
    X_new = []
    Y_new = []
    patient_id_list_new = []
    W_or_M_new = []
    for i,y in enumerate(Y):
        # if np.isnan(y):
        #     print("y is nan")
        if y == 'nan':
            #print("y is 'nan' ")
            continue
        else:
            Y_new.append(y)
            X_new.append(X[i])
            patient_id_list_new.append(patient_id_list[i])
            W_or_M_new.append(W_or_M[i])
    X = X_new
    Y = Y_new
    patient_id_list = patient_id_list_new
    W_or_M = W_or_M_new
    # print("In handleMissingData(). Xnew: ", X)
    # print("In handleMissingData(). Ynew: ", Y)

    return X,Y,patient_id_list, W_or_M

def generateOverlapping15mers(mutations):
    mut_to_wt15merList_dict = {}
    mut_to_mt15merList_dict = {}

    for mut in mutations:

        if mut == 'D614G':
            mut_to_wt15merList_dict[mut] = ['GTNTSNQVAVLYQDV','NQVAVLYQDVNCTEV','LYQDVNCTEVPVAIH','DVNCTEVPVAIHADQ']
            mut_to_mt15merList_dict[mut] = ['GTNTSNQVAVLYQGV','NQVAVLYQGVNCTEV','LYQGVNCTEVPVAIH','GVNCTEVPVAIHADQ']

        if mut == 'A570D':
            mut_to_wt15merList_dict[mut] = ['NKKFLPFQQFGRDIA','PFQQFGRDIADTTDA','GRDIADTTDAVRDPQ','ADTTDAVRDPQTLEI']
            mut_to_mt15merList_dict[mut] = ['NKKFLPFQQFGRDID','PFQQFGRDIDDTTDA','GRDIDDTTDAVRDPQ','DDTTDAVRDPQTLEI']

        if mut == 'P681H':
            mut_to_wt15merList_dict[mut] = ['CASYQTQTNSPRRAR','TQTNSPRRARSVASQ','PRRARSVASQSIIAY']
            mut_to_mt15merList_dict[mut] = ['CASYQTQTNSHRRAR','TQTNSHRRARSVASQ','HRRARSVASQSIIAY']

        if mut == 'S982A':
            mut_to_wt15merList_dict[mut] = ['GAISSVLNDILSRLD','VLNDILSRLDKVEAE','LSRLDKVEAEVQIDR']
            mut_to_mt15merList_dict[mut] = ['GAISSVLNDILARLD','VLNDILARLDKVEAE', 'LARLDKVEAEVQIDR']

        if mut == 'N501Y':
            mut_to_wt15merList_dict[mut] = ['PLQSYGFQPTNGVGY','GFQPTNGVGYQPYRV','NGVGYQPYRVVVLSF']
            mut_to_mt15merList_dict[mut] = ['PLQSYGFQPTYGVGY','GFQPTYGVGYQPYRV', 'YGVGYQPYRVVVLSF']

        if mut == 'T716I':
            mut_to_wt15merList_dict[mut] = ['AYSNNSIAIPTNFTI','SIAIPTNFTISVTTE','TNFTISVTTEILPVS']
            mut_to_mt15merList_dict[mut] = ['AYSNNSIAIPINFTI','SIAIPINFTISVTTE', 'INFTISVTTEILPVS']

        if mut == 'D1118H':
            mut_to_wt15merList_dict[mut] = ['QRNFYEPQIITTDNT','EPQIITTDNTFVSGN','TTDNTFVSGNCDVVI']
            mut_to_mt15merList_dict[mut] = ['QRNFYEPQIITTHNT','EPQIITTHNTFVSGN', 'TTHNTFVSGNCDVVI']

        if mut == 'K417N':
            mut_to_wt15merList_dict[mut] = ['EVRQIAPGQTGKIAD','APGQTGKIADYNYKL','GKIADYNYKLPDDFT']
            mut_to_mt15merList_dict[mut] = ['EVRQIAPGQTGNIAD','APGQTGNIADYNYKL', 'GNIADYNYKLPDDFT']

        if mut == 'E484K':
            mut_to_wt15merList_dict[mut] = ['EIYQAGSTPCNGVEG','GSTPCNGVEGFNCYF','NGVEGFNCYFPLQSY']
            mut_to_mt15merList_dict[mut] = ['EIYQAGSTPCNGVKG','GSTPCNGVKGFNCYF', 'NGVKGFNCYFPLQSY']

        if mut == 'A701V':
            mut_to_wt15merList_dict[mut] = ['SIIAYTMSLGAENSV','TMSLGAENSVAYSNN','AENSVAYSNNSIAIP']
            mut_to_mt15merList_dict[mut] = ['SIIAYTMSLGVENSV','TMSLGVENSVAYSNN', 'VENSVAYSNNSIAIP']

        if mut == 'L18F':
            mut_to_wt15merList_dict[mut] = ['VLLPLVSSQCVNLTT','VSSQCVNLTTRTQLP','VNLTTRTQLPPAYTN']
            mut_to_mt15merList_dict[mut] = ['VLLPLVSSQCVNFTT','VSSQCVNFTTRTQLP', 'VNFTTRTQLPPAYTN']

        if mut == 'T20N&L18F':
            mut_to_wt15merList_dict[mut] = ['VLLPLVSSQCVNLTT','VSSQCVNLTTRTQLP','VNLTTRTQLPPAYTN']
            mut_to_mt15merList_dict[mut] = ['VLLPLVSSQCVNFTN','VSSQCVNFTNRTQLP', 'VNFTNRTQLPPAYTN']

        if mut == 'T20N':
            mut_to_wt15merList_dict[mut] = ['VLLPLVSSQCVNLTT','VSSQCVNLTTRTQLP','VNLTTRTQLPPAYTN']
            mut_to_mt15merList_dict[mut] = ['VLLPLVSSQCVNLTN','VSSQCVNLTNRTQLP', 'VNLTNRTQLPPAYTN']

        if mut == 'P26S':
            mut_to_wt15merList_dict[mut] = ['VNLTTRTQLPPAYTN','RTQLPPAYTNSFTRG','PAYTNSFTRGVYYPD']
            mut_to_mt15merList_dict[mut] = ['VNLTTRTQLPSAYTN','RTQLPSAYTNSFTRG', 'SAYTNSFTRGVYYPD']

        if mut == 'D138Y':
            mut_to_wt15merList_dict[mut] = ['VVIKVCEFQFCNDPF','CEFQFCNDPFLGVYY','CNDPFLGVYYHKNNK']
            mut_to_mt15merList_dict[mut] = ['VVIKVCEFQFCNYPF','CEFQFCNYPFLGVYY', 'CNYPFLGVYYHKNNK']

        if mut == 'R190S':
            mut_to_wt15merList_dict[mut] = ['LMDLEGKQGNFKNLR','GKQGNFKNLREFVFK','FKNLREFVFKNIDGY', 'REFVFKNIDGYFKIY']
            mut_to_mt15merList_dict[mut] = ['LMDLEGKQGNFKNLS','GKQGNFKNLSEFVFK', 'FKNLSEFVFKNIDGY','SEFVFKNIDGYFKIY']

        if mut == 'K417T':
            mut_to_wt15merList_dict[mut] = ['EVRQIAPGQTGKIAD','APGQTGKIADYNYKL','GKIADYNYKLPDDFT']
            mut_to_mt15merList_dict[mut] = ['EVRQIAPGQTGTIAD','APGQTGTIADYNYKL', 'GTIADYNYKLPDDFT']

        if mut == 'T1027I':
            mut_to_wt15merList_dict[mut] = ['AEIRASANLAATKMS','SANLAATKMSECVLG','ATKMSECVLGQSKRV']
            mut_to_mt15merList_dict[mut] = ['AEIRASANLAAIKMS','SANLAAIKMSECVLG', 'AIKMSECVLGQSKRV']

        if mut == 'H655Y':
            mut_to_wt15merList_dict[mut] = ['NVFQTRAGCLIGAEH','RAGCLIGAEHVNNSY','IGAEHVNNSYECDIP', 'HVNNSYECDIPIGAG']
            mut_to_mt15merList_dict[mut] = ['NVFQTRAGCLIGAEY','RAGCLIGAEYVNNSY', 'IGAEYVNNSYECDIP','YVNNSYECDIPIGAG']

        if mut == 'D80A':
            mut_to_wt15merList_dict[mut] = ['HAIHVSGTNGTKRFD','SGTNGTKRFDNPVLP','TKRFDNPVLPFNDGV', 'DNPVLPFNDGVYFAS']
            mut_to_mt15merList_dict[mut] = ['HAIHVSGTNGTKRFA','SGTNGTKRFANPVLP', 'TKRFANPVLPFNDGV','ANPVLPFNDGVYFAS']

        if mut == 'D215G':
            mut_to_wt15merList_dict[mut] = ['FKIYSKHTPINLVRD','KHTPINLVRDLPQGF','NLVRDLPQGFSALEP', 'DLPQGFSALEPLVDL']
            mut_to_mt15merList_dict[mut] = ['FKIYSKHTPINLVRG','KHTPINLVRGLPQGF', 'NLVRGLPQGFSALEP','GLPQGFSALEPLVDL']

        if mut == 'R246I':
            mut_to_wt15merList_dict[mut] = ['TRFQTLLALHRSYLT','LLALHRSYLTPGDSS','RSYLTPGDSSSGWTA']
            mut_to_mt15merList_dict[mut] = ['TRFQTLLALHISYLT','LLALHISYLTPGDSS', 'ISYLTPGDSSSGWTA']

        if mut == '∆242-244':
            mut_to_wt15merList_dict[mut] = ['IGINITRFQTLLALH','TRFQTLLALHRSYLT','LLALHRSYLTPGDSS']
            mut_to_mt15merList_dict[mut] = ['IGINITRFQTLHRSY','TRFQTLHRSYLTPGD', 'LHRSYLTPGDSSSGW']


    return mut_to_wt15merList_dict, mut_to_mt15merList_dict


def generatePoolDicts():
    # # Hardcoded data from Cansu's excel speardsheet:
    # # Vaccinated cohort summary of mutation deconvolutions for predictions_updated

    pool_list_patIDs = [8001,8009,8052,9001,9003,9005,9006,9008,9012,9017,9019,9020]
    alphaPeptideIDs = [1,2,3,4,5,6,7]
    alphaMutations = ['D614G','A570D','P681H','S982A','N501Y','T716I','D1118H']

    betaPeptideIDs = [8,9,10,19,20,21,22]
    betaMutations = ['K417N','E484K','A701V','D80A','D215G','R246I','∆242-244']

    gammaPeptideIDs = [11,12,13,14,15,16,17,18,23]
    gammaMutations = ['L18F','T20N&L18F','P26S','D138Y','R190S','K417T','T1027I','H655Y','T20N']


    #Alpha (UK( Pool IFNg+ CD4+ T cells
    # WTWT=[17.30667,3.7,1.615,1.821,0.535,5.275,0.45,8.266667,2.986667,2.526667,0.300667,16.64]
    # WTMUT=[16.20667,3.82,0.965,0.691,0.375,4.185,0,6.156667,0.016667,1.896667,0,15.64]
    # MUTMUT=[18.5066667,5.6,0.485,0.491,0.765,9.845,0,2.93666667,0.53666667,3.09666667,0.22066667,14.14]
    # alphaCD4 = {}
    # alphaCD4['WTWT'] = WTWT
    # alphaCD4['WTMUT'] = WTMUT
    # alphaCD4['MUTMUT'] = MUTMUT


    #Alpha (UK)Pool IFNg+ CD8+ T cells
    WTWT= [0,1.12,0.425,0,0.265,1.89,0.43,0,1.05,1.386667,0,9.9775]
    WTMUT=[0,1.39,0,0,0.655,1.2,0,0.353333,0.16,1.616667,0.341667,10.3775]
    MUTMUT=[0.3,1.94,0.515,0.2,1.095,0.8,0.9,2.603333333,0.72,2.176666667,0.761666667,3.4775]
    MUTWT=[0,1.31,0,0,0.195,0.6,0,0,0,2.516666667,0.391666667,2.4575]
    alphaCD8 = {}
    alphaCD8['WTWT'] = {}
    alphaCD8['WTMUT'] = {}
    alphaCD8['MUTMUT'] = {}
    alphaCD8['MUTWT'] = {}
    for i, patID in enumerate(pool_list_patIDs):
        alphaCD8['WTWT'][patID] = WTWT[i]
        alphaCD8['WTMUT'][patID] = WTMUT[i]
        alphaCD8['MUTMUT'][patID] = MUTMUT[i]
        alphaCD8['MUTWT'][patID] = MUTWT[i]

    # #Beta (SA) Pool	IFNg+ CD4+ T cells
    # WTWT=18.40667,4.55,0,14.291,0.395,6.82,0.5,7.116667,14.38667,0.666667,0.070667,8.26
    # WTMUT=[8.366667,3.22,0,1.311,0.205,4.12,0.18,3.376667,4.336667,0.496667,0.180667,2.12]
    # MUTMUT=[10.2066667,3.61,0.325,0.531,0.965,19.18,0,4.75666667,10.1866667,0.53666667,0.11066667,0]
    # betaCD4 = {}
    # betaCD4['WTWT'] = WTWT
    # betaCD4['WTMUT'] = WTMUT
    # betaCD4['MUTMUT'] = MUTMUT

    #Beta (SA) Pool IFNg+ CD8+ T cells
    WTWT=[0.993333,0.19,0.2,0.143333,6.165,1.72,0.21,0.373333,0.98,1.616667,0.091667,1.8975]
    WTMUT=[0.283333,0.48,0,0,4.555,0.98,0.24,0.443333,1.08,1.866667,0.011667,0.7375]
    MUTMUT=[1.563333333,1.64,0,0,0.055,0.97,2.03,1.753333333,3.44,0.816666667,0,'nan']
    MUTWT=[0,0.08,0,0,0,1.47,0,0,0.51,0.376666667,0,0]
    betaCD8 = {}
    betaCD8['WTWT'] = {}
    betaCD8['WTMUT'] = {}
    betaCD8['MUTMUT'] = {}
    betaCD8['MUTWT'] = {}
    for i, patID in enumerate(pool_list_patIDs):
        betaCD8['WTWT'][patID] = WTWT[i]
        betaCD8['WTMUT'][patID] = WTMUT[i]
        betaCD8['MUTMUT'][patID] = MUTMUT[i]
        betaCD8['MUTWT'][patID] = MUTWT[i]

    # #Gamma (Brazil) Pool2 IFNg+ CD4+ T cells
    # WTWT=[6.826667,4.67,2.395,1.721,1.225,18.245,0.54,10.84667,17.18667,3.746667,0.310667,7.07]
    # WTMUT=[2.846667,2.9,0,0.371,1.145,6.595,0.9,4.726667,9.356667,2.476667,0.070667,7.2]
    # MUTMUT=[16.3066667,3.97,0.025,1.001,0.885,21.145,0,4.94666667,6.56666667,1.44666667,0,2.73]
    # gammaCD4 = {}
    # gammaCD4['WTWT'] = WTWT
    # gammaCD4['WTMUT'] = WTMUT
    # gammaCD4['MUTMUT'] = MUTMUT

    #Gamma (Brazil) Pool2 IFNg+ CD8+ T cells
    WTWT=[2.613333,1.01,0,0.32,0.685,1.04,1.89,3.393333,0.31,2.426667,0.221667,0]
    WTMUT=[1.073333,1.17,0,0,0.555,0.26,0.53,2.353333,0.76,1.646667,0,0]
    MUTMUT=[2.683333333,1.35,0,0,0.165,1.06,0,6.353333333,0.68,1.036666667,0.021666667,1.1975]
    MUTWT=[1.583333333,0.98,0,0,0,1.01,0,0.923333333,0,0.236666667,0.151666667,0]
    gammaCD8 = {}
    gammaCD8['WTWT'] = {}
    gammaCD8['WTMUT'] = {}
    gammaCD8['MUTMUT'] = {}
    gammaCD8['MUTWT'] = {}
    for i, patID in enumerate(pool_list_patIDs):
        gammaCD8['WTWT'][patID] = WTWT[i]
        gammaCD8['WTMUT'][patID] = WTMUT[i]
        gammaCD8['MUTMUT'][patID] = MUTMUT[i]
        gammaCD8['MUTWT'][patID] = MUTWT[i]

    return pool_list_patIDs, alphaPeptideIDs, alphaMutations, betaPeptideIDs, betaMutations , gammaPeptideIDs, gammaMutations, alphaCD8, betaCD8, gammaCD8



def generatePairwiseDistDict(mut_to_wt15merList_dict, mut_to_mt15merList_dict):

    len15_wt_sequences = []
    len15_mt_sequences = []
    # # -----------------------------------------------------------------------
    # # Precompute and save all pairwise dists btw 9mers.
    cansu_mt_9mers = []
    cansu_wt_9mers = []
    # # Iterate over mutations
    for mut in mut_to_mt15merList_dict:

        for s in mut_to_wt15merList_dict[mut]:
            len15_wt_sequences.append(s)
            cansu_wt_9mers.extend(generate_9_mers(s))

        for s in mut_to_mt15merList_dict[mut]:
            len15_mt_sequences.append(s)
            cansu_mt_9mers.extend(generate_9_mers(s))

    cansu_wt_9mers = list( set(cansu_wt_9mers))
    cansu_mt_9mers = list( set(cansu_mt_9mers))

    ninemer_pairwise_dist_dict = {}
    ninemer_pairwise_dist_dict_hamming = {}
    full_ninemer_list = list( set(cansu_wt_9mers+cansu_mt_9mers ))
    print("Number of 9mers/ no. pairs: ", len(full_ninemer_list),len(full_ninemer_list)*len(full_ninemer_list))
    for s1 in full_ninemer_list:

        # # Initialize ninemer_pairwise_dist_dict[s1] = {}
        ninemer_pairwise_dist_dict[s1] = {}
        ninemer_pairwise_dist_dict_hamming[s1] = {}
        for s2 in full_ninemer_list:

            # # Initialize ninemer_pairwise_dist_dict[s2] = {}
            if s2 not in ninemer_pairwise_dist_dict:
                ninemer_pairwise_dist_dict[s2] = {}
                ninemer_pairwise_dist_dict_hamming[s2] = {}

            if s1 == s2:
                ninemer_pairwise_dist_dict[s1][s2] = 0
                ninemer_pairwise_dist_dict[s2][s1] = 0

                ninemer_pairwise_dist_dict_hamming[s1][s2] = 0
                ninemer_pairwise_dist_dict_hamming[s2][s1] = 0
            else:
                d = dist(['',s1],['',s2])
                ninemer_pairwise_dist_dict[s1][s2] = d
                ninemer_pairwise_dist_dict[s2][s1] = d

                d_hamming = hamming(list(s1),list(s2)) * len(s1)
                ninemer_pairwise_dist_dict_hamming[s1][s2] = HFun(d_hamming)
                ninemer_pairwise_dist_dict_hamming[s2][s1] = HFun(d_hamming)

    return ninemer_pairwise_dist_dict, ninemer_pairwise_dist_dict_hamming

def generate_mutations_and_sequences(dfs):
    # # Look at Peptides
    peptide_df = dfs['Peptide sequences for mutations']

    # # Get Cansu sequences
    long_mt_sequences = list(peptide_df['Mutant Sequence'])
    long_wt_sequences = list(peptide_df['WT Sequence '])

    mutations = list(peptide_df['Mutation'])
    long_wt_mt_pair__to__mut_dict = {}
    for i,wt in enumerate(long_wt_sequences):
        mt = long_mt_sequences[i]
        long_wt_mt_pair__to__mut_dict[(wt,mt)] = mutations[i]

    return mutations, long_wt_mt_pair__to__mut_dict

def determine_Cp_dict(X, Y, Patients, model):
    '''
    See overleaf writeup. These are analytical solutions of del LogL / del cp == 0

    '''

    Cp_dict = {}
    unique_PatIds = list(set(Patients))
    if not model['Cp_specific']:
        # # Best cp is determined by applying [del LogL / del cp == 0] solution
        # # over all patients - i.e. use full X,Y arrays.
        cp = sum(np.array(Y)*np.array(X)) / sum(np.array(X)*np.array(X))
        for patID in unique_PatIds:
            Cp_dict[patID] = cp
    else:
        # # Best cp is determined by applying [del LogL / del cp == 0] solution
        # # for each patient separately.
        for patID in unique_PatIds:
            Xpat = np.array( [X[i] for i in range(len(X)) if Patients[i] == patID] )
            Ypat = np.array( [Y[i] for i in range(len(Y)) if Patients[i] == patID] )
            cp = sum(Ypat*Xpat) / sum(Xpat*Xpat)
            Cp_dict[patID] = cp

    return Cp_dict

def create_self_dicts(preT_dict_key, preT_dict_val):
    tpp = time.time()
    cutoffs_orig = [0,1,2,3,4,5,6,7,8,9,10]
    item = preT_dict_key

    # # CONSTRUCT HAMMING DICTS
    # # ##### This takes a few hours on macbook pro #####
    EpiHLA_to_H_to_SelfEpis = {}
    EpiHLA_to_H_to_SelfSethnaDists = {}
    EpiHLA_to_H_to_SelfKdsInv = {}

    EpiHLA_to_H_to_SelfEpis[item] = {}
    EpiHLA_to_H_to_SelfSethnaDists[item] = {}
    EpiHLA_to_H_to_SelfKdsInv[item] = {}

    for h in cutoffs_orig:
        EpiHLA_to_H_to_SelfEpis[item][h] = []
        EpiHLA_to_H_to_SelfSethnaDists[item][h] = []
        EpiHLA_to_H_to_SelfKdsInv[item][h] = []

     # # Iterate over self-epitopes for current item=(assay epi,hla)
    for index,hepi in enumerate(preT_dict_val[2]):
        assayepi = item[0]
        h = hamming([ch for ch in hepi],[ch for ch in assayepi])*9

        EpiHLA_to_H_to_SelfEpis[item][h].append(hepi)
        EpiHLA_to_H_to_SelfSethnaDists[item][h].append(preT_dict_val[0][index])
        EpiHLA_to_H_to_SelfKdsInv[item][h].append(preT_dict_val[1][index])

    # print("runtime create_self_dicts(): ", time.time()-tpp, flush=True)
    return EpiHLA_to_H_to_SelfEpis, EpiHLA_to_H_to_SelfSethnaDists, EpiHLA_to_H_to_SelfKdsInv


# #  ------------------ BEGIN POOL STIM, PEPTIDE RESTIM -----------------------
# # Load Cansu data
# dfs = pd.read_excel('/Users/marcus/Documents/GitHub/Cansu_COVID/Vaccinated cohort summary _12-09-22_MarcusFormat.xlsx', sheet_name=None, engine='openpyxl')
dfs = pd.read_excel('/mnt/scratch/marcust/WorkingDir/Cansu_COVID/Vaccinated cohort summary _12-09-22_MarcusFormat.xlsx', sheet_name=None, engine='openpyxl')

# # hla_seqq_dict: key=hla, value=list of 9-mers
# # -covers ALL HLAS, not just those of patients in cohort [needed to compute Kds for shufflings]
# # patientID_to_seq_hla_dict: key=patientID, value=list of [seq,hla] for the patient
tphl = time.time()
patientID_to_seq_hla_dict, hla_seqq_dict = generate_seq_hla_Info_SPIKE(dfs)
print("\n Time to generate seq,hla info: ", time.time() - tphl)

# # Generate experimental results
tpri = time.time()
pepID_EXCEL_COL = 'NORMALIZED' # 'FILTERED'
CD8_response_EXCEL_TAB = 'CD8+ response IFNg+ '+pepID_EXCEL_COL #'CD8+ response IFNg and TNFa+ '+pepID_EXCEL_COL
ResponseInfo, PatientID_PepID_Dict, PeptideID_to_Sequence_Dict = generatePatientResponseInfo_SPIKE(dfs, patientID_to_seq_hla_dict, CD8_response_EXCEL_TAB, pepID_EXCEL_COL)
print("\n Time to generate experimental response info: ", time.time() - tpri)

# # run netMHC on all (not just from overlapping 15mers for simplicity) 9mers
# tempdir='/mnt/scratch/marcust/WorkingDir/Cansu_COVID/testing2'#'/Users/marcus/Documents/Conifold_Storage/Cansu/testing'
# tempdir = os.path.join(tempdir,str(int(random.random()*100000)) )
# if not os.path.exists(tempdir):
#     os.makedirs(tempdir)

# tpkd = time.time()
# use_netMHC34 = False
# use_netMHC40 = True
# seqq_hla_to_Kd_dict = make_seqq_hla_to_Kd_dict(hla_seqq_dict, tempdir, use_netMHC34=use_netMHC34, use_netMHC40=use_netMHC40)
# print("\n Time to generate Kd(seq,hla) across ALL hlas (not just patient hlas): ", time.time() - tpkd)


# #  ----------------- BEGIN POOL STIM, POOL RESTIM ---------------------------
mutations, long_wt_mt_pair__to__mut_dict = generate_mutations_and_sequences(dfs)
# # Cansu actually used overlapping 15mers for "full coverage" of the
# # length ~25 sequences. However, not all 9mers generatable from the
# # long peptide will appear in the set of 9mers generated from the
# # overlapping 15mers.
# # mut_to_wt15merList_dict[mut] = [15mers]
mut_to_wt15merList_dict, mut_to_mt15merList_dict = generateOverlapping15mers(mutations)
ninemer_pairwise_dist_dict, ninemer_pairwise_dist_dict_hamming = generatePairwiseDistDict(mut_to_wt15merList_dict, mut_to_mt15merList_dict )


# # Retrieve Experimental Data for pool stimulation->pool restimulation
pool_list_patIDs, alphaPeptideIDs, alphaMutations, betaPeptideIDs, betaMutations , gammaPeptideIDs, gammaMutations, alphaCD8, betaCD8, gammaCD8 = generatePoolDicts()
# #  ------------------- END POOL STIM, POOL RESTIM ---------------------------


def eval_Obj_Function(params):
    # # Use max() aggrFun, no vaccine context best params as defualts
    if 'BetaParam' in params:
        BetaParam = params['BetaParam']
    else:
        BetaParam = 0.0
    if 'BetaParam_IEDB' in params:
        BetaParam_IEDB = params['BetaParam_IEDB']
    else:
        BetaParam_IEDB = 0.0

    if 'a_soft' in params:
        a_soft = params['a_soft']
    else:
        a_soft = 1
    if 'conc' in params:
        conc = params['conc']
    else:
        conc = 82200

    if 'conc_vaccine' in params:
        conc_vaccine = params['conc_vaccine']
    else:
        conc_vaccine = 1
    if 'sigma' in params:
        sigma = params['sigma']
    else:
        sigma = 1.147
    Cp_dict = {}

    global gammat
    if 'gammat' in params:
        gammat = params['gammat']
    else:
        gammat = 2.25e5

    global K0
    if 'K0' in params:
        K0 = params['K0']
    else:
        K0 = 500




    # # Compute Tolerances at current (gammat,K0)
    tpof = time.time()
    global T_dict
    T_dict = {}
    '''
    Update: now I define T_dict to be a global variable and update it in
    eval_ninemer_func(). Why? Time to create T_dict here was:
        len(preT_dict):  127716
        Time to compute T_dict:  1533.753173828125
    I think not all of its entries are actually used so better to only create
    the entries that are needed, as they are needed.
    '''


    # # ------  EVALUATE NEGATIVE LOGL AT CURRENT TOLERANCE PARAMS  -------
    # # For storing ninemer qualities from which effective number of
    # # 9mers per aggregation (based on entropy) will be computed.
    Q_allPools = {}

    X_allPools = []
    Y_allPools = []
    Patients_allPools = []
    PoolLabels = []
    WT_vs_MT_list = []
    # # Alpha, Beta, Gamma pools  ---------------------------------------------
    PoolDictList = [alphaCD8, betaCD8, gammaCD8 ]
    PoolMutList = [alphaMutations, betaMutations, gammaMutations ]
    for pl, poolDict in enumerate(PoolDictList):
        poolMutations = PoolMutList[pl] #gammaMutations

        # # The 9mers corresponding to each wt/mt long
        # # peptide actually derive from the overlapping
        # # 15mers associated with each long peptide.
        # # (i.e., not all 9mers derivable from the long
        # # peptide were actually represented in the
        # # overlapping 15mer set)
        # # compute_pool_ninemers() handles this.
        wm_pool_ninemers, ww_pool_ninemers, mm_pool_ninemers, mw_pool_ninemers, all_ninemers_set = compute_pool_ninemers(poolMutations, long_wt_mt_pair__to__mut_dict, mut_to_wt15merList_dict, mut_to_mt15merList_dict)

        # # Compute the min distances to IEDB for each 9mer.
        # # Let eval_Obj_Function() know this is a global variable so we can
        # # modify it (rather than just access it)
        global min_d_IEDB_dict
        global run_compute_min_d_IEDB_dict
        global run_compute_min_d_IEDB_counter
        if run_compute_min_d_IEDB_dict:
            tpci = time.time()
            min_d_IEDB_dict = compute_min_d_IEDB_dict( min_d_IEDB_dict, list(all_ninemers_set) )
            print("Time to compute min_d_IEDB_dict / pl / len(min_d_IEDB_dict): ", time.time() - tpci, pl, len(min_d_IEDB_dict), flush=True)
            print("min_d_IEDB_dict = ",min_d_IEDB_dict)
            run_compute_min_d_IEDB_counter += 1
            if run_compute_min_d_IEDB_counter>= 4:
                # # compute_min_d_IEDB_dict() should be called 3 times, one for each pool.
                run_compute_min_d_IEDB_dict = False

        if include_PoolStim_PoolRestim:
            Q_DICT, X, Y, patient_id_list, W_or_M = compute_XY_poolStim_poolRestim(aggrFun_ninemers,
                    a_soft, poolDict, poolMutations, patientID_to_seq_hla_dict,
                    seqq_hla_to_Kd_dict, long_wt_mt_pair__to__mut_dict, BetaParam, BetaParam_IEDB,
                    ninemer_pairwise_dist_dict,
                    ninemer_pairwise_dist_dict_hamming,
                    ww_pool_ninemers, mm_pool_ninemers, wm_pool_ninemers, mw_pool_ninemers,
                    conc, conc_vaccine,
                    include_WTWT=include_WTWT, include_WTMUT=include_WTMUT,
                    include_MUTMUT=include_MUTMUT,include_MUTWT=include_MUTWT)#, T_dict=T_dict)

            # # Only betaCD8 contains empty data fields.
            if poolDict == betaCD8:
                X,Y,patient_id_list, W_or_M = handleMissingData(X,Y,patient_id_list, W_or_M)

            # print("X = ", X)

            X_allPools.extend(X)
            Y_allPools.extend(Y)
            WT_vs_MT_list.extend(W_or_M)
            Patients_allPools.extend(patient_id_list)
            PoolLabels.extend([pl]*len(patient_id_list))
            Q_allPools[pl] = Q_DICT

        if include_PoolStim_PeptideRestim:
            w_pool_w_peptide_ninemers, m_pool_m_peptide_ninemers = compute_pool_peptide_ninemers(poolMutations, long_wt_mt_pair__to__mut_dict, mut_to_wt15merList_dict, mut_to_mt15merList_dict)

            Q_DICT, X, Y, patient_id_list, W_or_M = compute_XY_poolStim_peptideRestim(aggrFun_ninemers,
                    a_soft, poolDict, ResponseInfo, poolMutations, patientID_to_seq_hla_dict,
                    seqq_hla_to_Kd_dict, long_wt_mt_pair__to__mut_dict, BetaParam, BetaParam_IEDB,
                    PeptideID_to_Sequence_Dict, ninemer_pairwise_dist_dict,
                    ninemer_pairwise_dist_dict_hamming,
                    w_pool_w_peptide_ninemers, m_pool_m_peptide_ninemers,
                    conc, conc_vaccine,
                    include_WTWT=include_WTWT, include_MUTMUT=include_MUTMUT)#, T_dict=T_dict)

            if poolDict == betaCD8:
                X,Y,patient_id_list, W_or_M = handleMissingData(X,Y,patient_id_list, W_or_M)

            X_allPools.extend(X)
            Y_allPools.extend(Y)
            WT_vs_MT_list.extend(W_or_M)
            Patients_allPools.extend(patient_id_list)
            PoolLabels.extend([pl]*len(patient_id_list))
            Q_allPools[pl] = Q_DICT
            # print("X: ", X)
            # print("max(X), max(X_allPools), max(Y), max(Y_allPools): ", max(X), max(X_allPools), max(Y), max(Y_allPools), flush=True)

    # # DETERMINE OPTIMAL Cp list if not provided
    if Cp_dict == {}:
        Cp_dict = determine_Cp_dict(X_allPools, Y_allPools, Patients_allPools, model)

    # # Apply patient specific factors cp to model values
    for i,x in enumerate(X_allPools):
        patID = Patients_allPools[i]
        cp = Cp_dict[patID]
        X_allPools[i] = cp*x



    #%% ----------   BEGIN ENTROPY SECTION   ---------------
    Entropy_Info = {}
    # # For the current aggrFun, we want to compute the entropy and perplexity
    # # over all pre-aggregation 9mer sets.
    # # This collection of entropy (perplexity) values can then be plotted.
    EP_DICT = {}
    ENTROPY_LIST = []
    PERPLEXITY_LIST = []
    for pl in Q_allPools:
        EP_DICT[pl] = {}

        for patID in Q_allPools[pl]:
            EP_DICT[pl][patID] = {}

            for hla in Q_allPools[pl][patID]:
                EP_DICT[pl][patID][hla] = {}

                for condition in Q_allPools[pl][patID][hla]:
                    # # condition: 'WTWT', 'WTMUT', etc.

                    ninemer_qualities_preAggr = Q_allPools[pl][patID][hla][condition]

                    # # Normalize 9mer qualitites to form a quality distribution
                    ninemer_qualities_preAggr /= sum(ninemer_qualities_preAggr)

                    # # Store entropy and perplexity (e^Entropy) of the
                    # # pre-aggregation ninemer distribution
                    E = entropy(ninemer_qualities_preAggr)
                    Perplexity = np.exp(E)
                    EP_DICT[pl][patID][hla][condition] = [E, Perplexity]

                    ENTROPY_LIST.append(E)
                    PERPLEXITY_LIST.append(Perplexity)


    Entropy_Info['perplexities'] = PERPLEXITY_LIST
    Entropy_Info['entropies'] = ENTROPY_LIST
    Entropy_Info['Q_allPools'] = Q_allPools
    #%% ----------   END ENTROPY SECTION   ---------------



    n = len(X_allPools)
    negLogLikelihood = (n*np.log(2*np.pi *(sigma**2))/2) + (1/(2*sigma*sigma))*sum( (np.array(X_allPools) - np.array(Y_allPools) )**2 )
    if math.isnan(negLogLikelihood):
        print("X_allPools-Y_allPools", np.array(X_allPools) - np.array(Y_allPools))
        print("X_allPools: ",X_allPools)
        print("sigma: ", sigma)
        print("Quitting")
        sys.exit()
    OBJVAL = negLogLikelihood


    BIC_Info = {}
    BIC_Info['sigma'] = sigma

    k = 0 # number of parameters
    if use_conc:
        BIC_Info['conc'] = conc
        k += 1

    if model['use_tolerance']:
        k += 3 # sigma, gammat, K0
        BIC_Info['gammat'] = gammat
        BIC_Info['K0'] = K0
    else:
        k += 1 # sigma

    if model['include_vaccine']:
        k += 1 # conc_vaccine
        BIC_Info['conc_vaccine'] = conc_vaccine

    if aggrFun_ninemers == 'realSoftMax' or aggrFun_ninemers == 'realSoftMin':
        k += 1 # a_soft
        BIC_Info['a_soft'] = a_soft

    if model['Cp_specific']:
        numPatParams = 13
        BIC_Info['Cp_dict'] = Cp_dict
    else:
        numPatParams = 1

    numBetaParams = 0
    if model['include_crossReactivity']:
        numBetaParams += 1
        BIC_Info['BetaParam'] = BetaParam
    if model['IEDB'] != 'No_IEDB':
        numBetaParams += 1
        BIC_Info['BetaParam_IEDB'] = BetaParam_IEDB




    k += numPatParams + numBetaParams


    LogLikelihood = -negLogLikelihood
    BIC = k*np.log(n) - (2*LogLikelihood)
    RMSD = np.sqrt( np.mean( (np.array(X_allPools) - np.array(Y_allPools))**2 ) )
    BIC_Info['BIC'] = BIC
    BIC_Info['k'] = k
    BIC_Info['n'] = n
    BIC_Info['LogL'] = LogLikelihood
    BIC_Info['RMSD'] = RMSD

    print("n / OBJFUN / Time to eval obj function: ", n, OBJVAL, time.time()-tpof, flush=True)

    return {"loss": OBJVAL, "status": STATUS_OK, "BIC_Info": BIC_Info, "Entropy_Info": Entropy_Info}


if __name__ == "__main__":
    tp0 = time.time()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ninemerFun", "--ninemerFun", help="data directory")
    parser.add_argument("-use_tolerance", "--use_tolerance", help="data directory")
    parser.add_argument("-include_vaccine", "--include_vaccine", help="data directory")
    parser.add_argument("-Cp_specific", "--Cp_specific", help="data directory")
    parser.add_argument("-include_crossReactivity", "--include_crossReactivity", help="data directory")
    parser.add_argument("-use_conc", "--use_conc", help="data directory")
    parser.add_argument("-max_evals", "--max_evals", help="data directory")
    parser.add_argument("-outdir", "--outdir", help="data directory")
    parser.add_argument("-HLA_reshuffle_number", "--HLA_reshuffle_number", help="")
    parser.add_argument("-randomizeHLAs", "--randomizeHLAs", help="")
    parser.add_argument("-IEDB", "--IEDB")

    args = parser.parse_args()
    ninemerFun = args.ninemerFun
    aggrFun_ninemers = ninemerFun

    random.seed(202)

    HLA_reshuffle_number = int(args.HLA_reshuffle_number)
    if args.randomizeHLAs == 'True':
        randomizeHLAs = True
    else:
        randomizeHLAs = False

    if args.use_conc == 'True':
        use_conc = True
    else:
        use_conc = False

    space = {}
    if ninemerFun == 'realSoftMax' or ninemerFun == 'realSoftMin':
        space["a_soft"] = hp.uniform("a_soft",0, 1000)

    space["conc"] = hp.loguniform("conc",np.log(1), np.log(100000))
    space["sigma"] = hp.uniform("sigma",1,100)

    model = {}
    if args.use_tolerance == 'True':
        model['use_tolerance'] = True
        # space["gammat"] = hp.loguniform("gammat", np.log(1e4), np.log(1e6)) # for delE = KdInv or dH*KdInv
        space["gammat"] = hp.loguniform("gammat", np.log(1e4), np.log(1e8)) # for delE = KdInv or KdInv*(1/dSethna)
        # space["gammat"] = hp.loguniform("gammat", np.log(1e-8), np.log(1e3)) # for delE = dist_Sethna
        #space["K0"] = hp.uniform("K0",100, 5000)
    else:
        model['use_tolerance'] = False

    if args.include_vaccine == 'True':
        model['include_vaccine'] = True
        space["conc_vaccine"] = hp.loguniform("conc_vaccine",np.log(1), np.log(100000))
    else:
        model['include_vaccine'] = False

    if args.Cp_specific == 'True':
        model['Cp_specific'] = True
    else:
        model['Cp_specific'] = False
    if args.include_crossReactivity == 'True':
        model['include_crossReactivity'] = True
        space["BetaParam"] = hp.loguniform("BetaParam", np.log(0.001), np.log(100))
    else:
        model['include_crossReactivity'] = False

    if args.IEDB == 'Everything':
        model['IEDB'] = 'Everything'
        min_d_IEDB_dict = {}
        run_compute_min_d_IEDB_dict = True
        run_compute_min_d_IEDB_counter = 1
        space["BetaParam_IEDB"] = hp.loguniform("BetaParam_IEDB", np.log(0.001), np.log(10))
    elif args.IEDB == 'No_Covid':
        model['IEDB'] = 'No_Covid'
        min_d_IEDB_dict = {}
        run_compute_min_d_IEDB_dict = True
        run_compute_min_d_IEDB_counter = 1
        space["BetaParam_IEDB"] = hp.loguniform("BetaParam_IEDB", np.log(0.001), np.log(10))
    elif args.IEDB == 'Only_Covid':
        model['IEDB'] = 'Only_Covid'
        min_d_IEDB_dict = {}
        run_compute_min_d_IEDB_dict = True
        run_compute_min_d_IEDB_counter = 1
        space["BetaParam_IEDB"] = hp.loguniform("BetaParam_IEDB", np.log(0.001), np.log(10))
    else:
        model['IEDB'] = 'No_IEDB'
        run_compute_min_d_IEDB_dict = False

    max_evals = int(args.max_evals)
    saveFile = True
    include_PoolStim_PeptideRestim = False
    include_PoolStim_PoolRestim = True

    include_WTWT = True
    include_WTMUT = True
    include_MUTMUT = True
    include_MUTWT = True
    include_first_stimulation = True


    print("include_WTWT, include_WTMUT, include_MUTMUT, include_MUTWT, include_first_stimulation: ", include_WTWT, include_WTMUT, include_MUTMUT, include_MUTWT, include_first_stimulation)
    print("model['IEDB'] = ",model['IEDB'])

    tempdir='/mnt/scratch/marcust/WorkingDir/Cansu_COVID/testing2'#'/Users/marcus/Documents/Conifold_Storage/Cansu/testing'
    tempdir = os.path.join(tempdir,str(int(random.random()*100000)) )
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)

    tpkd = time.time()
    use_netMHC34 = False
    use_netMHC40 = True
    if randomizeHLAs:
        print("Computing (randomized HLAs) Kd info...")
        seqq_hla_to_Kd_dict = make_seqq_hla_to_Kd_dict_randomizedHLAs(hla_seqq_dict, tempdir, HLA_reshuffle_number, use_netMHC34=use_netMHC34, use_netMHC40=use_netMHC40)
    else:
        print("Computing (true HLAs) Kd info...")
        seqq_hla_to_Kd_dict = make_seqq_hla_to_Kd_dict(hla_seqq_dict, tempdir, use_netMHC34=use_netMHC34, use_netMHC40=use_netMHC40)
    print("\n Time to generate Kd(seq,hla) across ALL hlas (not just patient hlas): ", time.time() - tpkd)


    rstate = np.random.default_rng(31415)
    trials = Trials()

    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    toldir='/mnt/scratch/marcust/WorkingDir/Cansu_COVID/Tolerance_Intermediates_SelfNeighborhoodSize_10000'
    loadfile = os.path.join(toldir,'SPIKE_9mers_L_H_allHLAs_preT_info.pkl')

    # # preT_dict is a global variable since it is not defined within a function
    with open(loadfile,'rb') as lf:
        preT_dict = pickle.load(lf)

    # # These dicts will hold the *relevant* preT_dict information partitioned
    # # by hamming distance.
    # # Not all entries in preT_dict will be used (since it was computed for all
    # # netMHC alleles)
    hamming_cutoff = 5
    EpiHLA_to_H_to_SelfEpis = {}
    EpiHLA_to_H_to_SelfSethnaDists = {}
    EpiHLA_to_H_to_SelfKdsInv = {}



    # # TO OPTIMIZE NEG LOG LIKELIHOOD
    best = fmin(
        eval_Obj_Function,
        space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=rstate,
        verbose=0
    )

    # # -------   BEGIN PLOT/SAVE ENTROPY INFO   --------------
    # # Retrieve Entropy_Info
    results = trials.results
    output_list = [item for item in results if item['loss'] == min(trials.losses())]
    print("output_list: ", output_list)
    Entropy_Info = output_list[0]['Entropy_Info']
    PERPLEXITY_LIST = Entropy_Info['perplexities']

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    indexList = [u for u in range(len(PERPLEXITY_LIST))]
    ax1.scatter(indexList,PERPLEXITY_LIST, marker ="o", s=8)

    mean_perplexity = sum(PERPLEXITY_LIST)/len(PERPLEXITY_LIST)
    std_perplexity = stats.stdev(PERPLEXITY_LIST)


    if include_PoolStim_PeptideRestim:
        title = "Perplexity values of pre-Aggregation 9-mer sets \n9mer sets -> (patient,peptide,hla,condition) \nMean "+str(round(mean_perplexity,1))+", Stdev "+str(round(std_perplexity,1))
    elif include_PoolStim_PoolRestim:
        title = "Perplexity values of pre-Aggregation 9-mer sets \n9mer sets -> (patient,pool,hla,condition) \nMean "+str(round(mean_perplexity,1))+", Stdev"+str(round(std_perplexity,1))
    xlab = "9-mer set index"
    ylab = "Perplexity (Eff. No. of 9-mers contributing to aggregation)"

    ax1.set_title(title)
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab)
    fig_file = os.path.join(outdir,'Perplexity_Plot_peptideRestim_'+str(include_PoolStim_PeptideRestim)+'_poolRestim_'+str(include_PoolStim_PoolRestim)+'_WTWT_'+str(include_WTWT)+'_WTMUT_'+str(include_WTMUT)+'_MUTMUT_'+str(include_MUTMUT)+'_MUTWT_'+str(include_MUTWT)+'.png')
    plt.savefig(fig_file, dpi=600, bbox_inches='tight')

    data_file = os.path.join(outdir,'Entropy_Info_peptideRestim_'+str(include_PoolStim_PeptideRestim)+'_poolRestim_'+str(include_PoolStim_PoolRestim)+'_WTWT_'+str(include_WTWT)+'_WTMUT_'+str(include_WTMUT)+'_MUTMUT_'+str(include_MUTMUT)+'_MUTWT_'+str(include_MUTWT)+'.pkl')
    with open(data_file,"wb") as pkd:
        pickle.dump(Entropy_Info, pkd)
    # # -------   END PLOT/SAVE ENTROPY INFO   --------------


    print(best)
    print("min trials.losses(): ",  min(trials.losses()))
    print("run time: ", time.time() - tp0)



    # # ------   DO NOT SAVE SEARCH RESULTS   ----------
    # bestLoss = min(trials.losses())
    # ParamSets_achieving_bestLoss = [item['misc']['vals'] for i,item in
    #                                 enumerate(trials.trials) if trials.losses()[i] == bestLoss]
    #
    # # # Just in case multiple param sets achieve the min obj value, store
    # # # BIC info for each.
    # results = [item for item in results if item['loss'] == min(trials.losses())]
    # print("results: ", results)
    #
    # outfile_base = os.path.join(outdir, ninemerFun+'_MLE_model')
    # if randomizeHLAs:
    #     suffix_randomizeHLAs = '_randomizeHLAs_'+str(HLA_reshuffle_number)
    # else:
    #     suffix_randomizeHLAs = ''
    #
    # if not include_WTWT or not include_MUTMUT or not include_WTMUT or not include_MUTWT:
    #     suffix_includeExper = '_include_WTWT_'+str(include_WTWT)+'_include_MUTMUT_'+str(include_MUTMUT)+'_include_WTMUT_'+str(include_WTMUT)+'_include_MUTWT_'+str(include_MUTWT)
    # else:
    #     suffix_includeExper = ''
    #
    # if not include_first_stimulation:
    #     suffix_first_stim = '_no_First_Stim'
    # else:
    #     suffix_first_stim = ''
    #
    # if model['IEDB'] != 'No_IEDB':
    #     suffix_IEDB = '_IEDB_'+model['IEDB']
    # else:
    #     suffix_IEDB = ''
    # outfile = outfile_base+suffix_randomizeHLAs+suffix_includeExper+suffix_first_stim+suffix_IEDB+'.txt'
    #
    # lines = ["ParamSet(s) Achieving Best loss: "+str(ParamSets_achieving_bestLoss)]
    # negLL = round(min(trials.losses()))
    # lines2 = ["Best loss (NegLogL): "+str(negLL)]
    #
    # # # BIC info
    # lines3 = []
    # for item in results:
    #     lines3.append(str(item))
    #
    # # # Runtime and numevals
    # lines4 = ["runtime: "+str(round(time.time()-tp0, 2))+", No. func evals: "+str(max_evals)]
    #
    #
    # if saveFile:
    #     # # Store MLE optimization results incl BIC
    #     with open(outfile, 'w') as f:
    #         for line in lines:
    #             f.write(line)
    #             f.write('\n')
    #         for line in lines2:
    #             f.write(line)
    #             f.write('\n')
    #         for line in lines3:
    #             f.write(line)
    #             f.write('\n')
    #         for line in lines4:
    #             f.write(line)
    #             f.write('\n')
    #
    #
    # self_dicts = [EpiHLA_to_H_to_SelfEpis, EpiHLA_to_H_to_SelfSethnaDists, EpiHLA_to_H_to_SelfKdsInv]
    # tpst = time.time()
    # with open(os.path.join(toldir,'SPIKE_9mers_self_dicts.pkl'),"wb") as psk:
    #     pickle.dump(self_dicts, psk)
    # print("time to save self_dicts pkl file: ", time.time()-tpst, flush=True)
