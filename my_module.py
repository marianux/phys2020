#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:08:31 2020

@author: mariano
"""
# type check libs

import numpy as np

from scipy import signal as sig
from numbers import Number
import pdb

import matplotlib.pyplot as plt

from scipy.signal._peak_finding_utils import (
    _select_by_peak_distance,
)


def my_int(x):
    
    return int(np.round(x))

def my_ceil(x):
    
    return int(np.ceil(x))


def zero_crossings( x ):

    zero_crossings = np.where(np.diff(np.signbit(x)))[0]

    return(zero_crossings)

def if_not_empty(x):

    if len(x) > 0:
        return(x)
    else:
        return(np.nan)

def keep_local_extrema(x, peaks, zero_crossings, distance  = None, filter_win = None):

    try:

        zero_crossings = np.array(zero_crossings).reshape([len(zero_crossings),1])
        zc = np.hstack([zero_crossings[0:-2], zero_crossings[1:-1], zero_crossings[2:]])
        # zc = np.hstack([zero_crossings[0:-1], zero_crossings[1:]])
    
        # pesamos cada cruce por cero de acuerdo a los máximos circundantes.
        local_extrema_weight = np.array([ np.sum(np.abs( if_not_empty(x[peaks[ np.logical_and( peaks > zc[ii,0], peaks < zc[ii,2])]])  )) for ii in range(zc.shape[0]) ])
    
        aux_idx = np.logical_not(np.isnan(local_extrema_weight)).nonzero()[0]
        zc = zc[aux_idx,1]
        local_extrema_weight = local_extrema_weight[aux_idx]
    
        # hago un filtrado de ventana para intentar no perder extrremos 
        # al final remuevo las redundancias
        extrema_keep = 10
        
        if isinstance(filter_win, type(None) ):
            filter_win = 500 # 1 second assuming 500 Hz sampling rate
            
        aux_idx = [ np.logical_and( jj <= zc, jj+filter_win > zc).nonzero()  for jj in np.arange(0, x.shape[0], filter_win//2) ]
        aux_idx2 = [ np.argsort(local_extrema_weight[jj])[-extrema_keep:] for jj in aux_idx ]
    
        try:
            # pad to extrema_keep length
            aux_idx2 = [ np.concatenate( [jj, [jj[-1]]*(extrema_keep - len(jj))]).astype(np.int) for jj in aux_idx2 ]
        except Exception as ex:
            print(ex)
            pdb.set_trace()
            raise(ex)
    
        # try:
        #     aux_idx3 = [local_extrema_weight[jj] for jj in aux_idx2 ]
        #     zc_r, aux_idx4 = np.unique( np.vstack([ zc[jj[0][kk]] for (jj,kk) in zip(aux_idx,aux_idx2) ]), return_index=True )
        # except:
        #     pdb.set_trace()
        
        aux_idx3 = [local_extrema_weight[jj] for jj in aux_idx2 ]
        zc_r, aux_idx4 = np.unique( np.vstack([ zc[jj[0][kk]] for (jj,kk) in zip(aux_idx,aux_idx2) ]), return_index=True )
        local_extrema_weight_r = np.vstack(aux_idx3).flatten()[aux_idx4]


        aux_idx = [ np.logical_and( jj <= peaks, jj+filter_win > peaks).nonzero()  for jj in np.arange(0, x.shape[0], filter_win//2) ]
        aux_idx2 = [ np.argsort(np.abs( if_not_empty(x[peaks[jj]] )))[-extrema_keep:] for jj in aux_idx ]

        try:
            # pad to extrema_keep length
            aux_idx2 = [ np.concatenate( [jj, [jj[-1]]*(extrema_keep - len(jj))]).astype(np.int) for jj in aux_idx2 ]
        except Exception as ex:
            print(ex)
            pdb.set_trace()
            raise(ex)

        peaks_r = np.unique( np.vstack([ peaks[jj[0][kk]] for (jj,kk) in zip(aux_idx,aux_idx2) ]))
    
    except Exception as ex:
        print(ex)
        pdb.set_trace()
        raise(ex)


# plt.figure(2); t_range=(10e3, 20e3); plt.clf(); plt.plot( np.arange(t_range[0], t_range[1]), x[int(t_range[0]):int(t_range[1])]); this_peaks = zc_r[ np.logical_and( zc_r > t_range[0],  zc_r < t_range[1] )]; plt.plot(this_peaks, x[this_peaks], 'ro' );  plt.pause(10)


    # filtramos cruces por cero más próximos que "distance"
    if isinstance(distance, type(None) ):
        local_zc = zc_r
        local_peaks = peaks_r
    else:
        keep = _select_by_peak_distance(zc_r, local_extrema_weight_r, distance)
        local_zc = zc_r[keep]
        
        keep = _select_by_peak_distance(peaks_r, np.abs(x[peaks_r]), distance)
        local_peaks = peaks_r

    return( local_zc, local_peaks )

def my_find_extrema( x, this_distance = None ):

    try:

        peaks_p = [ sig.find_peaks(np.squeeze(x[:,jj]), distance = this_distance )[0]  for jj in range(x.shape[1]) ]
        peaks_n = [ sig.find_peaks(-np.squeeze(x[:,jj]), distance = this_distance )[0]  for jj in range(x.shape[1]) ]
        zeros = [ zero_crossings( np.squeeze(x[:,jj]) ) for jj in range(x.shape[1]) ]
    
        # a partir de los picos y cruces, me quedo con los cruces más importantes.
        my_extrema = [ keep_local_extrema(x[:,ii], np.sort( np.hstack([jj,kk])), ll, this_distance)  for (ii, jj, kk, ll) in zip(range(x.shape[1]), peaks_p, peaks_n, zeros ) ]
        
        my_zc = [ my_extrema[ii][0] for ii in range(x.shape[1]) ]
        my_ex = [ my_extrema[ii][1] for ii in range(x.shape[1]) ]
    
        
    except Exception as ex:
        print(ex)
        pdb.set_trace()
        raise(ex)

    return(my_zc, my_ex)


def sync_sig(sig, sync_seq, t_win ):

    synced_marks = np.array([])
    
    if  np.all([isinstance(ii, Number) for ii in t_win]):
        pre_win, post_win = t_win
    else:
        pre_win = 1000
        post_win = 1000
        
    the_marks = np.arange(sig.shape[0])
    sync_sig = np.stack([ sig[np.logical_and( the_marks > (ii-pre_win), the_marks < (ii+post_win)), :]  for ii in sync_seq ], axis = 2)
    
    return(sync_sig)


def sync_marks(sync_seq, the_marks, t_win ):
    
    synced_marks = np.array([])
    
    if  np.all([isinstance(ii, Number) for ii in t_win]):
        pre_win, post_win = t_win
    else:
        pre_win = 1000
        post_win = 1000
        
    the_marks = np.array(the_marks)
    grouped_mark_idx = [ np.logical_and( the_marks > (ii-pre_win), the_marks < (ii+post_win)).nonzero()[0] for ii in sync_seq ]
    grouped_mark = np.hstack([ the_marks[this_mark_idx] - ii + pre_win - 1 for (ii,this_mark_idx) in zip(sync_seq, grouped_mark_idx) ])
    mark_dist = np.hstack([ np.abs(the_marks[this_mark_idx] - ii) for (ii,this_mark_idx) in zip(sync_seq, grouped_mark_idx) ])
    grouped_mark_idx = np.hstack(grouped_mark_idx).flatten()
    
    aux_idx = np.argsort(grouped_mark_idx)
    grouped_mark_idx = grouped_mark_idx[aux_idx]
    mark_dist = mark_dist[aux_idx]
    grouped_mark = grouped_mark[aux_idx]
    
    for ii in (np.diff(grouped_mark_idx) == 0).nonzero()[0]:
    
        aux_idx = (grouped_mark_idx == grouped_mark_idx[ii]).nonzero()[0]
        
        aux_idx2 = np.argsort(mark_dist[aux_idx])
        
        grouped_mark[ aux_idx[aux_idx2[1:]]] = -1
        grouped_mark_idx[ aux_idx[aux_idx2[1:]]] = -1
        
    grouped_mark = grouped_mark[grouped_mark != -1]
    grouped_mark_idx = grouped_mark_idx[grouped_mark_idx != -1]
    
    return(grouped_mark, the_marks[grouped_mark_idx])


def plot_ecg_mosaic( data,  qrs_locations = None, target_lead_names = None, ecg_header = None,  t_win = None, row_cols = None, marks = None):
    """
    Plot a multichannel or multilead ECG signal in multiple panels. 
    Each channel can be synchronized to a qrs_location time reference. 
    Text and marks can be also included.

    Call signatures::

        plot_ecg_mosaic( data )

    Parameters
    ----------
    data : { numpy 3D matrix (samples, channels, other ... ) }
        The tridimensional data matrix must have time in the first dimension,
        channels in the second, and eventually realizations or repetitions 
        of each channel in the third dimension. The figure will contain a panel 
        for each channel, and within each panel, as many series or lines as the 
        third dimension size.


    qrs_locations : {None, numeric sequence}, optional
        The sample index to data used as synch, where at each sample in 
        qrs_locations, a window defined with t_win is considered to be plotted 
        in each panel. See also t_win.

    ecg_header : { None or dict}, optional
        Description of the ECG typically available in the header. Dictionary 
        with fields:

          -freq: Sampling rate in Hz. (1)
          -sig_len: Number of ECG samples. (data.shape[0])
          -n_sig: Number of ECG leads. (data.shape[1])
          -adczero: ADC offset (e.g. 2^(adc_res-1) when
          using unsigned integers). ( repmat(0, ECG_header.nsig , 1) ) 
          -adcgain: ADC gain in units/adc_sample
          (typically uV/adc_unit). ( repmat(1, ECG_header.nsig , 1) )
          -units: Measurement units. ( repmat('uV', ECG_header.nsig , 1) )
          -desc: Signal description. ( num2str(colvec(1:ECG_header.nsig)) )

    target_lead_names : { None, string or numeric index}, optional

    t_win : { None, numeric}, optional

    Returns
    -------
    axes : `~.axes.Axes` (or a subclass of `~.axes.Axes`)
        The returned axes class depends on the projection used. It is
        `~.axes.Axes` if rectilinear projection are used and
        `.projections.polar.PolarAxes` if polar projection
        are used.

    Notes
    -----
    Notes.

    See Also
    --------
    .Figure.add_axes
    .pyplot.subplot
    .pyplot.subplots

    Examples
    --------
    ::

        #Creating a new full window axes
        plt.axes()

        #Creating a new axes with specified dimensions and some kwargs
        plt.axes((left, bottom, width, height), facecolor='w')
    """
    try:
        if isinstance(ecg_header, type(None) ):
            
            ecg_header = {}
            ecg_header['sig_len'] , ecg_header['n_sig'] = data.shape
            ecg_header['sig_name'] = np.arange(ecg_header['n_sig'])
            ecg_header['fs'] = 500
            
        if isinstance(target_lead_names, type(None) ):
            # all signals default
            target_lead_names = ecg_header['sig_name']
            
        if  np.all([isinstance(ii, Number) for ii in t_win]):
            pre_win, post_win = t_win 
        else:
            # all data length covered
            pre_win = my_int( ecg_header['fs'] * 0.3 )
            post_win = my_int( ecg_header['fs'] * 0.5 )
        
        if  np.all([isinstance(ii, Number) for ii in row_cols]):
            nrows, ncols = row_cols 
        else:
            # all data length covered
            ncols = 3
            nrows = my_ceil( ecg_header['n_sig'] / ncols )
    
        [_, target_lead_idx, _] = np.intersect1d(ecg_header['sig_name'], target_lead_names,  assume_unique=True, return_indices=True)
        
        fig = plt.figure(figsize = [11.69, 8.27])
        axs = fig.subplots(nrows, ncols, sharex='col', sharey='row',
                                gridspec_kw={'hspace': 0, 'wspace': 0})

        if isinstance(qrs_locations, type(None) ):
            # no qrs provided, a simulated default location at rec start
            qrs_locations = 0
        else:
            qrs_locations = qrs_locations[ np.logical_and(qrs_locations > pre_win, qrs_locations < (ecg_header['sig_len'] - post_win) ) ]
        
        for (jj, this_ax) in zip(target_lead_idx, axs.flat):

            # plot signals             
            sync_beats = np.array([ data[ ii-pre_win:ii+post_win, jj] - np.median(data[ ii-pre_win:ii+post_win, jj])  for ii in qrs_locations ]).transpose()
            
            this_ax.plot( sync_beats )

            if not isinstance(marks, type(None) ):
                # plot marks if available
                this_marks = np.array(marks[jj])
                sync_marks = np.unique(np.hstack([ this_marks[ np.logical_and( this_marks > ii-pre_win, this_marks < ii+post_win)] - ii + pre_win for ii in qrs_locations ]))
                    
                if len(sync_marks) > 0:
                    y_lim = this_ax.get_ylim()
                    this_ax.plot( [sync_marks] * 2, np.array([y_lim] * len(sync_marks)).transpose(), 
                             color='gray', marker='|', linestyle=':',
                             linewidth=1, markersize=7 ) 
    
            
            this_ax.text(0.05, 0.92, ecg_header['sig_name'][jj], style='italic',
                    bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 6}, transform=this_ax.transAxes)
    
            # sb_bar = np.median(sync_beats, axis=1, keepdims=True)
            # sb_mad = np.median( np.abs(sync_beats - sb_bar) )
            # str_mad = '{:f}'.format(sb_mad)
    
        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(0, pre_win + post_win, ecg_header['fs'] * 0.1, dtype='int' )
        minor_ticks = np.arange(0, pre_win + post_win, ecg_header['fs'] * 0.02, dtype='int' )
        
        major_ticks_lab = np.arange(-pre_win, post_win, ecg_header['fs'] * 0.1, dtype='int' )
            
        for this_ax in axs.flat:
            this_ax.label_outer()
            
            this_ax.set_xticks( major_ticks )
            this_ax.set_xticks( minor_ticks, minor=True)
    
            # this_ax.set_xticklabels( [str(atick) for atick in major_ticks_lab] )
            this_ax.set_xticklabels( major_ticks_lab )
    
            # And a corresponding grid
            this_ax.grid(which='x')
            
            # Or if you want different settings for the grids:
            this_ax.grid(which='minor', alpha=0.2)
            this_ax.grid(which='major', alpha=0.5)

    except Exception as ex:
        print(ex)
        pdb.set_trace()
        raise(ex)
        
    return(fig)
    

# if __name__ == '__main__':
    # por si hubiera que inicializar algo
    