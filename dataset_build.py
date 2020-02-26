#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:21:20 2020

@author: mariano
"""

import numpy as np
import sys
from fractions import Fraction
import pandas as pd
import os
from glob import glob
#import h5py
import wfdb as wf
from scipy import signal as sig
import scipy.io as sio

import argparse as ap
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb



def get_records( db_path, db_name ):

    all_records = []
    all_patient_list = []
    size_db = []

    for this_db in db_name:

        records = []
        patient_list = []

        # particularidades de cada DB
        if this_db == 'mitdb':
            
            records = ['100', '101', '103', '105', '106', '108', '109', '111', '112', '113', '114', '115', '116', '117', '118', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234']
            patient_list = np.arange(0, len(records)) + 1
            
            
            
        elif this_db == 'svdb':
            records = ['800', '801', '802', '803', '804', '805', '806', '807', '808', '809', '810', '811', '812', '820', '821', '822', '823', '824', '825', '826', '827', '828', '829', '840', '841', '842', '843', '844', '845', '846', '847', '848', '849', '850', '851', '852', '853', '854', '855', '856', '857', '858', '859', '860', '861', '862', '863', '864', '865', '866', '867', '868', '869', '870', '871', '872', '873', '874', '875', '876', '877', '878', '879', '880', '881', '882', '883', '884', '885', '886', '887', '888', '889', '890', '891', '892', '893', '894']
            patient_list = np.arange(0, len(records)) + 1
            
        elif this_db == 'INCART':
            # INCART: en esta DB hay varios registros por paciente
            records = [ 'I01', 'I02', 'I03', 'I04', 'I05', 'I06', 'I07', 'I08', 'I09', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16', 'I17', 'I18', 'I19', 'I20', 'I21', 'I22', 'I23', 'I24', 'I25', 'I26', 'I27', 'I28', 'I29', 'I30', 'I31', 'I32', 'I33', 'I34', 'I35', 'I36', 'I37', 'I38', 'I39', 'I40', 'I41', 'I42', 'I43', 'I44', 'I45', 'I46', 'I47', 'I48', 'I49', 'I50', 'I51', 'I52', 'I53', 'I54', 'I55', 'I56', 'I57', 'I58', 'I59', 'I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69', 'I70', 'I71', 'I72', 'I73', 'I74', 'I75']
    
            patient_list = np.array([
                             1, 1 ,    # patient 1
                             2, 2, 2 , # patient 2
                             3, 3 ,    # ...
                             4 ,       #
                             5, 5, 5 , #
                             6, 6, 6 , #
                             7 , #
                             8, 8 , #
                             9, 9 , #
                             10, 10, 10 , #
                             11, 11 , #
                             12, 12 , #
                             13, 13 , #
                             14, 14, 14, 14 , #
                             15, 15 , #
                             16, 16, 16 , #
                             17, 17 , #
                             18, 18 , #
                             19, 19 , #
                             20, 20, 20 , #
                             21, 21 , #
                             22, 22 , #
                             23, 23, 23 , #
                             24, 24, 24 , #
                             25, 25 , #
                             26, 26, 26 , #
                             27, 27, 27 , #
                             28, 28, 28 , #
                             29, 29 , #
                             30, 30 , #
                             31, 31 , # ...
                             32, 32   # patient 32
                             ])
        else:
            
            # para el resto de dbs, un registro es un paciente.                        
            data_path = os.path.join(db_path, this_db)
            
            paths = glob(os.path.join(data_path, '*.hea'))
        
            if paths == []:
                continue
                
            # Elimino la extensión
            paths = [os.path.split(path) for path in paths]
            file_names = [path[1][:-4] for path in paths]
            file_names.sort()
            
            records = file_names
            patient_list = np.arange(0, len(records)) + 1


        print( 'Procesando ' + this_db )

        records = [os.path.join(this_db, this_rec) for this_rec in records ]
        
        size_db += [ len(np.unique(patient_list)) ]
        all_records += records
        all_patient_list = np.hstack( [ all_patient_list, (patient_list + len(np.unique(all_patient_list)) ) ] )


    return all_records, all_patient_list, size_db


def make_dataset(records, data_path, ds_config, leads_x_rec = [], data_aumentation = 1, ds_name = 'none'):


    # Recorro los archivos
#    for this_rec in records:
    
    if len(leads_x_rec) == 0:
        leads_x_rec = ['all'] * len(records) 
       
    
    all_signals = []
    all_extrema = []
    ds_part = 1
    cant_total_samples = []
    parts_samples = 0
    parts_recordings = 0
    ds_parts_fn = []


    # image folder
    image_path = os.path.join( ds_config['dataset_path'], 'images' )

    qrs_det_path = os.path.join('.', 'qrs_detections')


    os.makedirs(image_path, exist_ok=True)

    target_columns = ['Age','Sex','Dx']

    target_classes = ['AF','I-AVB','LBBB','Normal','PAC','PVC','RBBB','STD','STE']

    df_empty = pd.DataFrame([], columns= target_columns + target_classes )
    df_all = df_empty

    start_beat_idx = 8

#    for this_rec in records:
    for ii in np.arange(start_beat_idx, len(records)):

        this_rec = records[ii]
        
        print ( str(ii) + ' - ' + str(my_int(ii / len(records) * 100)) + '% Procesando:' + this_rec)
        
        data, ecg_header = wf.rdsamp(os.path.join(data_path, this_rec) )

        # parse diagnostics
        this_rec_name = this_rec.split('/')[1]
        parsed_comments = np.array([re.search('(.*)\:\ (.*)', aa).group(1, 2) for aa in ecg_header['comments'] ]).transpose()
        df_aux = pd.DataFrame( parsed_comments[1,:], index= parsed_comments[0,:], columns= [this_rec_name] ).T
        this_df = df_empty 
        this_df = this_df.append( df_aux[target_columns])
        this_df[target_classes] = False
        present_diagnostic = this_df['Dx'].item().split(',')
        # turn on boolean target classes
        this_df[present_diagnostic] = True
        
        # construct a dataframe with target data
        df_all = df_all.append(this_df)

        # pq_ratio = ds_config['target_fs']/ecg_header['fs']
        # resample_frac = Fraction( pq_ratio ).limit_denominator(20)
        # #recalculo el ratio real
        # pq_ratio = resample_frac.numerator/ resample_frac.denominator
        # data = sig.resample_poly(data, resample_frac.numerator, resample_frac.denominator )
        # ecg_header['sig_len'] = data.shape[0]
            
        # data = median_baseline_removal( data, fs = ds_config['target_fs'])
# x_st = 200000; x_off = 10000; plt.plot(data[x_st:x_st+x_off,:]); plt.show()
        
#        plt.figure(1); plt.plot(data); plt.plot([0, data.shape[0]], [k_conv, k_conv], 'r:'); plt.plot([0, data.shape[0]], [-k_conv, -k_conv], 'r:'); this_axes = plt.gca(); this_axes.set_ylim([-(2**15-1), 2**15-1]); plt.show(1);

        # load QRS detections
        mat_struct = sio.loadmat( os.path.join( qrs_det_path, this_rec_name + '_QRS_detections.mat' ) )

        qrs_locs = np.vstack(mat_struct['mixartif_ECGmix1']['time'][0]).flatten()


        rec_sz = ecg_header['sig_len']
        half_rec = my_int(rec_sz/2)-500
        win_sz = 1000
        gap_sz = 50
        start_idx = np.array([0, half_rec, rec_sz-win_sz])
        face_colors = [(0.7, 0.2, 0.2, 0.3), (0.2, 0.2, 0.7, 0.3), (0.2, 0.7, 0.2, 0.3)]
        xaxis_idx = np.arange(0, len(start_idx)) * win_sz
        xgap = gap_sz * np.arange(0,len(xaxis_idx))
        plt.figure(1);
        plt.clf()
        
        [ plt.plot( np.arange(xaxis_idx[ii], xaxis_idx[ii]+win_sz) + xgap[ii], data[start_idx[ii]:start_idx[ii]+win_sz,:] )  for ii in range(len(start_idx)) ]
        
        ax = plt.gca()
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

        for ii in range(len(start_idx)):
            
            this_locs = qrs_locs[ np.logical_and( qrs_locs > start_idx[ii], qrs_locs < (start_idx[ii]+win_sz)) ] - start_idx[ii] + xaxis_idx[ii] + xgap[ii]
            plt.plot( [this_locs] * 2, np.array([y_lim] * len(this_locs)).transpose(), 'v:k' ) 
            
            rect = patches.Rectangle((xaxis_idx[ii]+ xgap[ii], y_lim[0]), win_sz, y_lim[1] - y_lim[0], linewidth=1, edgecolor='k',facecolor=face_colors[ii])
            ax.add_patch(rect)
            

        plt.title( this_rec_name + '- Dx: ' + this_df['Dx'].item() )
        
        plt.savefig( os.path.join( image_path, this_rec_name + '.jpg'), dpi=150)

        pre_win = my_int( ecg_header['fs'] * 0.3 )
        post_win = my_int( ecg_header['fs'] * 0.5 )
        
        target_lead_names =  ['II', 'V1']
    
        [_, target_lead_idx, _] = np.intersect1d(ecg_header['sig_name'], target_lead_names,  assume_unique=True, return_indices=True)
        
        
        for jj in target_lead_idx:
            plt.figure(1);
            plt.clf()
            
            qrs_locs = qrs_locs[ np.logical_and(qrs_locs > pre_win, qrs_locs < (rec_sz - post_win) ) ]
            
            sync_beats = np.array([ data[ ii-pre_win:ii+post_win, jj] - np.median(data[ ii-pre_win:ii+post_win, jj])  for ii in qrs_locs ]).transpose()
            sb_bar = np.median(sync_beats, axis=1, keepdims=True)
            sb_mad = np.median( np.abs(sync_beats - sb_bar) )
            
            plt.plot( sync_beats )
    
            str_mad = '{:f}'.format(sb_mad)
            plt.title( this_rec_name + '- Dx: ' + this_df['Dx'].item() + ' - mad: ' + str_mad )
    
            plt.savefig( os.path.join( image_path, str_mad + this_rec_name + '_sync' + ecg_header['sig_name'][jj] + '.jpg'), dpi=150)


    #     all_signals += [data]
    #     all_extrema += [rel_extrema]
    #     parts_samples += data.shape[0]
    #     parts_recordings += 1

    #     if sys.getsizeof(all_signals) > ds_config['dataset_max_size']:
            
    #         part_fn =  'ds_' + ds_name +  '_part_' + str(ds_part) + '.npy'

    #         ds_parts_fn += [ part_fn ]
    #         cant_total_samples += [parts_samples]
    #         cant_total_recordings += [parts_recordings]
             
    #         np.save( os.path.join( ds_config['dataset_path'], part_fn),  {'signals' : all_signals,
    #                                                                       'rel_extrema' : all_extrema,
    #                                                                       'lead_names'  : default_lead_order})
    #         ds_part += 1
    #         all_signals = []
    #         parts_samples = 0

    # if ds_part > 1 :
    #     # last part
        
    #     part_fn =  'ds_' + ds_name +  '_part_' + str(ds_part) + '.npy'

    #     ds_parts_fn += [ part_fn ]
    #     cant_total_samples += [parts_samples]
    #     cant_total_recordings += [parts_recordings]
             
    #     np.save( os.path.join( ds_config['dataset_path'], part_fn),  {'signals' : all_signals, 'rel_extrema' : all_extrema, 'lead_names'  : default_lead_order , 'cant_total_samples' : cant_total_samples})
    #     all_signals = []
        
    #     aux_df = pd.DataFrame( { 'filename': ds_parts_fn, 
    #                           'ds_samples': cant_total_samples,
    #                           'ds_recs': cant_total_recordings
    #                           } )
        
    # else:
        
    #     part_fn =  'ds_' + ds_name + '.npy'
    #     # unique part
    #     np.save( os.path.join( ds_config['dataset_path'], part_fn),  {'signals' : all_signals, 'rel_extrema' : all_extrema, 'lead_names'  : default_lead_order , 'cant_total_samples' : cant_total_samples})

    #     aux_df = pd.DataFrame( { 'filename': [part_fn], 
    #                           'ds_samples': [parts_samples],
    #                           'ds_recs': [parts_recordings]
    #                            } )
    
    # aux_df.to_csv( os.path.join(ds_config['dataset_path'], ds_name + '_size.txt'), sep=',', header=False, index=False)

    # algunos arreglos de cómo vinieron los datos
    df_all['Age'][ df_all['Age'].str.lower() == 'nan' ] = '0'
    df_all[ df_all['Age'].str.startswith('-') ]['Age']
    df_all['Age'] = pd.to_numeric(df_all['Age'])
    df_all[target_classes] = df_all[target_classes].convert_dtypes()

    df_all.to_csv( os.path.join(ds_config['dataset_path'], ds_name + '_data.txt'), sep=',', header=True, index=True)
    np.save( os.path.join(ds_config['dataset_path'], ds_name + '_dataframe'),  {'df_phys2020_training' : df_all})

def my_int(x):
    
    return int(np.round(x))

def my_ceil(x):
    
    return int(np.ceil(x))


########################
### Start of script ####
########################
    

parser = ap.ArgumentParser(description='Test of the first ideas for Physionet2020')
parser.add_argument( 'db_path', 
                     default='/home/mariano/mariano/dbs/', 
                     type=str, 
                     help='Path a la base de datos')

parser.add_argument( '--db_name', 
                     default=None, 
                     nargs='*',
                     type=str, 
                     help='Nombre de la base de datos')

parser.add_argument( '--particion', 
                     default=None, 
                     nargs='+',
                     type=float, 
                     help='Cantidad de pacientes en el set de training-val-test')

args = parser.parse_args()

db_path = args.db_path
db_name = args.db_name
# tamaño fijo del train, el resto val y test 50% each
partition = args.particion    # patients

known_db_names = ['physionet2020_training']

aux_df = None

if db_name is None:
    # default databases
    db_name = known_db_names
        
    # Train-val-test
    partition_mode = '3way'

else :
    
    if len(db_name) == 1 and os.path.isfile(db_name[0]):
        # El dataset lo determina un archivo de configuración externo.
        # Train-val-test
        partition_mode = '3way'
        
        # aux_df = pd.read_csv(db_name[0], header= 0, names=['rec', 'lead'])
        
    else:
        
        db_name_found = np.intersect1d(known_db_names, db_name)
        
        if len(db_name_found) == 0:
            
            print('No pude encontrar: ' + str(db_name) )
            sys.exit(1)

        db_name = db_name_found.tolist()
        
        if partition is None:
            # Esquemas para el particionado de los datos:
            # DB completa, db_name debería ser una 
            partition_mode = 'WholeDB'
        else:
            partition_mode = '3way'

    
#if not type(db_name) != 'list':
#    # force a list        
#    
#    db_name = [db_name]


cp_path = os.path.join('.', 'checkpoint')
os.makedirs(cp_path, exist_ok=True)

dataset_path = os.path.join('.', 'datasets')
os.makedirs(dataset_path, exist_ok=True)
#dataset_path = '/tmp/datasets/'

result_path = os.path.join('.', 'results')
os.makedirs(result_path, exist_ok=True)



ds_config = { 
                'width': 10, # s
                
                'target_fs':        200, # Hz

                'heartbeat_width': .09, # (s) Width of the type of heartbeat to seek for
                'distance':  .3, # (s) Minimum separation between consequtive QRS complexes
                'explore_win': 60, # (s) window to seek for possible heartbeats to calculate scales and offset

                'max_prop_3w_x_db': [0.8, 0.1, 0.1], # máxima proporción para particionar cada DB 
                
                'data_div_train': os.path.join(dataset_path, 'data_div_train.txt'),
                'data_div_val':   os.path.join(dataset_path, 'data_div_val.txt'),
                'data_div_test':  os.path.join(dataset_path, 'data_div_test.txt'),
                
                'dataset_max_size':  800*1024**2, # bytes
#                'dataset_max_size':  3e35, # infinito bytes
#                'target_beats': 2000, # cantidad máxima de latidos por registro
                'target_beats': None, # cantidad máxima de latidos por registro
                    
                'dataset_path':   dataset_path,
                'results_path':   result_path,
                
                'train_filename': os.path.join(dataset_path, 'train_' + '_'.join(db_name) + '.npy'),
                'test_filename':  os.path.join(dataset_path, 'test_' + '_'.join(db_name) + '.npy'),
                'val_filename':   os.path.join(dataset_path, 'val_' + '_'.join(db_name) + '.npy')
             } 

bForce_data_div = True
#bForce_data_div = False

if partition_mode == 'WholeDB':

    bForce_data_div = True

else:
    
    # 3-Way split
    if np.sum( np.array(partition) ) == 1 :
        # proportions
        tgt_train_size = partition[0] # patients
        if( len(partition) > 1 ) :
            tgt_val_size = partition[1] # patients
            if( len(partition) > 2 ) :
                tgt_test_size = partition[2] # patients
            else:
                tgt_test_size = (1-tgt_train_size-tgt_val_size) # patients
        else :
            tgt_test_size = (1-tgt_train_size)/2 # patients
            tgt_val_size = (1-tgt_train_size)/2 # patients
        
    else:
        
        # absolute values
        tgt_train_size = partition[0] # patients
        
        if( len(partition) > 1 ) :
            tgt_val_size = partition[1] # patients
            if( len(partition) > 2 ) :
                tgt_test_size = partition[2] # patients
            else:
                tgt_test_size = 0 # patients
        else :
            tgt_val_size = 0 # patients
            tgt_test_size = 0 # patients
        

#if  not os.path.isfile( ds_config['train_filename'] ) or bRedo_ds:

if bForce_data_div or not os.path.isfile( ds_config['data_div_train'] ):

    # reviso las db
    record_names, patient_list, size_db = get_records(db_path, db_name)
    leads_x_rec = ['all'] * len(record_names)
        
        
    # debug
    #record_names = record_names[0:9]

    patient_indexes = np.unique(patient_list)
    cant_patients = len(patient_indexes)
#    record_names = np.unique(record_names)
    cant_records = len(record_names)


    print('\n')
    aux_str = 'Bases de datos analizadas:'
    print( aux_str )
    print( '#' * len(aux_str) )
    [ print('{:s} : {:d} pacientes'.format(this_db, this_size)) for (this_db, this_size) in zip(db_name, size_db)]
    
    print('\n\n')
    aux_str = 'TOTAL: {:d} pacientes y {:d} registros.'.format(cant_patients, cant_records)
    print( '#' * len(aux_str) )
    print( aux_str )
    print( '#' * len(aux_str) )
    
    if partition_mode == 'WholeDB':
    
        db_start = np.hstack([ 0, np.cumsum(size_db[:-1]) ])
        db_end = db_start + size_db
        db_idx = np.hstack([np.repeat(ii, size_db[ii]) for ii in range(len(size_db))])
        
        
        for jj in range(len(db_name)):
            
            aux_idx = (db_idx == jj).nonzero()
            train_patients = np.sort(np.random.choice(patient_indexes[aux_idx], size_db[jj], replace=False ))
            
            aux_idx = np.hstack([ (patient_list==pat_idx).nonzero() for pat_idx in train_patients]).flatten()
            this_db_recs = [record_names[my_int(ii)] for ii in aux_idx]
    
            np.savetxt( os.path.join(ds_config['dataset_path'], db_name[jj] + '_recs.txt') , this_db_recs, '%s')

            print( 'Construyendo dataset ' + db_name[jj] )
            print( '#######################################' )
        
            # Armo el set de entrenamiento, aumentando para que contemple desplazamientos temporales
            signals, labels, ds_parts = make_dataset(this_db_recs, db_path, ds_config, ds_name = db_name[jj], data_aumentation = 1 )
            
        
    elif partition_mode == '3way':
        # propocion de cada db en el dataset
        cant_pacientes = np.sum(size_db)
        
        if tgt_train_size <= 1 and tgt_train_size >= 0 :
            tgt_train_size = my_int(cant_pacientes * tgt_train_size)
        
        if tgt_val_size <= 1 and tgt_val_size >= 0 :
            tgt_val_size = my_int(cant_pacientes * tgt_val_size)
            
        if tgt_test_size <= 1 and tgt_test_size >= 0 :
            tgt_test_size = my_int(cant_pacientes * tgt_test_size)
            
        print('\n')
        aux_str = 'Construyendo datasets ' ' Train: {:3.0f} - Val: {:3.0f} - Test: {:3.0f}'.format(tgt_train_size, tgt_val_size, tgt_test_size)            
        print( aux_str )
        print( '#' * len(aux_str) )
            
        prop_db = size_db / cant_pacientes
        
#        # particionamiento de 3 vías 
#        # train      80%
#        # validation 10%
#        # eval       10%
#        tgt_train_size = my_int(cant_patients * 0.8)
        
        
        # proporciones del corpus completo
#        tgt_db_parts_size = tgt_train_size * prop_db
#        tgt_db_parts_size = [ my_int(ii) for ii in tgt_db_parts_size]

        tgt_train_db_parts_size = tgt_train_size * np.ones(len(prop_db)) / len(prop_db)
        tgt_train_db_parts_size = [ my_ceil(ii) for ii in tgt_train_db_parts_size]
        
        tgt_val_db_parts_size = tgt_val_size * np.ones(len(prop_db)) / len(prop_db)
        tgt_val_db_parts_size = [ my_ceil(ii) for ii in tgt_val_db_parts_size]
        
        tgt_test_db_parts_size = tgt_test_size * np.ones(len(prop_db)) / len(prop_db)
        tgt_test_db_parts_size = [ my_ceil(ii) for ii in tgt_test_db_parts_size]
        
        db_start = np.hstack([ 0, np.cumsum(size_db[:-1]) ])
        db_end = db_start + size_db
        db_idx = np.hstack([np.repeat(ii, size_db[ii]) for ii in range(len(size_db))])
        
        train_recs = []
        val_recs = []
        test_recs = []
        
        train_leads_x_rec = []
        val_leads_x_rec = []
        test_leads_x_rec = []
        
        max_prop_3w_x_db = ds_config['max_prop_3w_x_db']
        
        for jj in range(len(db_name)):
            
            aux_idx = (db_idx == jj).nonzero()
            np.random.randint(len(aux_idx))
            train_patients = np.sort(np.random.choice(patient_indexes[aux_idx], np.min( [my_int(size_db[jj]*max_prop_3w_x_db[0]),  tgt_train_db_parts_size[jj] ]), replace=False ))
            test_patients = np.sort(np.setdiff1d(patient_indexes[aux_idx], train_patients, assume_unique=True))
            # test y val serán la misma cantidad de pacientes
            val_patients = np.sort(np.random.choice(test_patients, np.min( [len(test_patients)-1, my_int(size_db[jj]*max_prop_3w_x_db[1]), tgt_val_db_parts_size[jj] ]), replace=False ))
            test_patients = np.setdiff1d(test_patients, val_patients, assume_unique=True)

            test_patients = np.sort(np.random.choice(test_patients, np.min( [len(test_patients), my_int(size_db[jj]*max_prop_3w_x_db[2]), tgt_test_db_parts_size[jj] ]), replace=False ))
            
            if len(train_patients) > 0 :
                aux_idx = np.hstack([ (patient_list==pat_idx).nonzero() for pat_idx in train_patients]).flatten()
                aux_val = [record_names[my_int(ii)] for ii in aux_idx]
                train_recs += aux_val
                
                aux_val = [leads_x_rec[my_int(ii)] for ii in aux_idx]
                train_leads_x_rec += aux_val
    
            if len(val_patients) > 0 :
                aux_idx = np.hstack([ (patient_list==pat_idx).nonzero() for pat_idx in val_patients]).flatten()
                aux_val = [record_names[my_int(ii)] for ii in aux_idx]
                val_recs += aux_val

                aux_val = [leads_x_rec[my_int(ii)] for ii in aux_idx]
                val_leads_x_rec += aux_val

            if len(test_patients) > 0 :
                aux_idx = np.hstack([ (patient_list==pat_idx).nonzero() for pat_idx in test_patients]).flatten()
                aux_val = [record_names[my_int(ii)] for ii in aux_idx]
                test_recs += aux_val
                
                aux_val = [leads_x_rec[my_int(ii)] for ii in aux_idx]
                test_leads_x_rec += aux_val
        
    #    # particionamiento de 3 vías 
    #    # train      80%
    #    # validation 20%
    #    # eval       20%
    #    train_recs = np.random.choice(record_names, int(cant_records * 0.8), replace=False )
    #    test_recs = np.setdiff1d(record_names, train_recs, assume_unique=True)
    #    val_recs = np.random.choice(train_recs, int(cant_records * 0.2), replace=False )
    #    train_recs = np.setdiff1d(train_recs, val_recs, assume_unique=True)
        
    #    data_path = os.path.join(db_path, db_name)
    
        np.savetxt(ds_config['data_div_train'], train_recs, '%s')
        np.savetxt(ds_config['data_div_val'], val_recs, '%s')
        np.savetxt(ds_config['data_div_test'], test_recs, '%s')
 
else:
        
    train_recs = np.loadtxt(ds_config['data_div_train'], dtype=str )
    val_recs = np.loadtxt(ds_config['data_div_val'], dtype=str)
    test_recs = np.loadtxt(ds_config['data_div_test'], dtype=str)


if partition_mode == '3way':
    
    if len(train_recs) > 0 :
    
        print('\n')
        print( 'Construyendo el train' )
        print( '#####################' )
    
        # Armo el set de entrenamiento, aumentando para que contemple desplazamientos temporales
        make_dataset(train_recs, db_path, ds_config, leads_x_rec = train_leads_x_rec, ds_name = 'train', data_aumentation = 1 )

    if len(val_recs) > 0 :
    
        print('\n')
        print( 'Construyendo el val' )
        print( '###################' )
        # Armo el set de validacion
        make_dataset(val_recs, db_path, ds_config, leads_x_rec = val_leads_x_rec, ds_name = 'val')

    if len(test_recs) > 0 :

        print('\n')
        print( 'Construyendo el test' )
        print( '####################' )
        # Armo el set de testeo
        make_dataset(test_recs, db_path, ds_config, leads_x_rec = test_leads_x_rec, ds_name = 'test')

