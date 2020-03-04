#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numbers
from numpy import max, abs, round, all, log10, pi, diff, array,  logspace, linspace,  exp, mean, finfo
from scipy.signal import freqz, remez

import matplotlib.pyplot as plt

def plot_response(xx, F):
    
    _,g_amp_real = freqz(xx, 1, F * pi); g_amp_real = 20*log10(abs(g_amp_real)); plt.plot(F, g_amp_real); plt.show()
    
    return

def find_idx(x):
    indexes = array(x.nonzero(), dtype='uint').flatten()
    return(indexes)

def qs_filter_design(scales = [4], fs = 250, N = None, debug = False):
    """
    (*ecg-kit internal*) Design the wavelet decomposition filters for wavedet algorithm

    Mimics the transfer function of the filters used for ECG delineation in
    Martinez2004_ for an arbitrary sampling frequency and filter order N.
    The original sampling rate is 250 Hz, this routine calculates a new set of 
    filters that mimics the original magnitude response.

    Parameters
    ----------
    scales : : *array_like, optional*
        The desired scales to calculate from from 1 to 6. *Default: 4*
    fs  : : *scalar, optional*
        The target frequency of the wavelet filters. *Default: 250*
    N  : : *int, optional*
        The order of the filters in case the estimation of the order fails to 
        converge. *Default: automatically estimated*
    debug : : *boolean, optional*
            A flag to display some extra info and frequency response of the
            designed filters. *Default: Falsef*


    Returns
    -------
    out : : *tuple*
        A tuple with the FIR differentiators for each scale defined in **scales**.

    See Also
    --------
    
    Examples
    --------
    >>> wt_filters = qs_filter_design( scales = np.arange(1,5), fs = 500, debug=True )
    
    >>> wt_filters = qs_filter_design( scales = 4, fs = 360)

    WARNING
    -------
    As this routines iterates through several configurations in order to
    converge, the user should check the transfer functions of the filters. My
    suggestion is once you obtain a desired filter bank for a given Fs or
    configuration, save or cache it in a .mat file in order to use it during
    operation. For example, if you usually work with signals sampled at 360
    Hz, a good choice is to have a cached version of the filters for this Fs
    in a .mat file called "wt_filters_6 scales_360 Hz.mat". You can use this
    function on-line with your algorithm at your own risk.

    References
    ----------
    .. [#Martinez2004] Martinez et al. "A Wavelet-Based ECG Delineator: Evaluation on Standard
           Databases" IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. 51, NO. 4,
           APRIL 2004.

    Author
    ------    
    :Author: Mariano Llamedo Soria (llamedom at frba.utn.edu.ar)
    :Version: 0.1 beta
    :Birthdate: 17/2/11 (Translated to Python on Wed Sep  4 10:57:51 2019)
    :Last update: 22/02/13
    :Copyright: 2008-2019

    """
    
    eps = finfo('float32').eps
    
    filter_count = 0
    q_filters = []

    design_iter_attemps = 10;
    
    if not isinstance(fs, numbers.Number) :
        fs = 250 # Hz

    
    # escalas entre 1 y 6 (0 y 5)
    if not all([ (isinstance(ii, numbers.Number) and ii > 0 and ii < 6 ) for ii in scales] ) :
        scales = array([4])-1 # Hz
    else:
        scales = array(scales)-1

    
    if( not isinstance(N, numbers.Number) ):
        # valor emp�rico obtenido de varios dise�os.
        N = max([10, round(fs*4/30 + 16+2/3)])

    
    # Pruebo que N no sea demasiado distinto a lo recomendable.
    recommended_N = max([10, round(fs*4/30 + 16+2/3)]);
    if( abs(N - recommended_N) > 0.1*N ):
        print( 'Check the transfer functions of the differentiator filters designed. Recommended order N = ' + str(recommended_N) )

    
    #frecuencia a la que est� dise�ado el delineador, y que se toma para
    #referencia para que las escalas signifiquen lo mismo a cualquier Fs.
    f_ref = 250 #Hz
    f_ratio = f_ref/fs
    
    
    # Funciones de transferencia correctas para el dise�o de los filtros utilizadas 
    # por el wavedet a 250 Hz.
    empirical_tf = [ 
                    [2, -2],
                    [0.250000000000000, 0.750000000000000, 0.500000000000000, -0.500000000000000, -0.750000000000000, -0.250000000000000, ], 
                    [0.0312500000000000, 0.0937500000000000, 0.187500000000000, 0.312500000000000, 0.343750000000000, 0.281250000000000, 0.125000000000000, -0.125000000000000, -0.281250000000000, -0.343750000000000, -0.312500000000000, -0.187500000000000, -0.0937500000000000, -0.0312500000000000, ], 
                    [0.00390625000000000, 0.0117187500000000, 0.0234375000000000, 0.0390625000000000, 0.0585937500000000, 0.0820312500000000, 0.109375000000000, 0.140625000000000, 0.160156250000000, 0.167968750000000, 0.164062500000000, 0.148437500000000, 0.121093750000000, 0.0820312500000000, 0.0312500000000000, -0.0312500000000000, -0.0820312500000000, -0.121093750000000, -0.148437500000000, -0.164062500000000, -0.167968750000000, -0.160156250000000, -0.140625000000000, -0.109375000000000, -0.0820312500000000, -0.0585937500000000, -0.0390625000000000, -0.0234375000000000, -0.0117187500000000, -0.00390625000000000, ], 
                    [0.000488281250000000, 0.00146484375000000, 0.00292968750000000, 0.00488281250000000, 0.00732421875000000, 0.0102539062500000, 0.0136718750000000, 0.0175781250000000, 0.0219726562500000, 0.0268554687500000, 0.0322265625000000, 0.0380859375000000, 0.0444335937500000, 0.0512695312500000, 0.0585937500000000, 0.0664062500000000, 0.0727539062500000, 0.0776367187500000, 0.0810546875000000, 0.0830078125000000, 0.0834960937500000, 0.0825195312500000, 0.0800781250000000, 0.0761718750000000, 0.0708007812500000, 0.0639648437500000, 0.0556640625000000, 0.0458984375000000, 0.0346679687500000, 0.0219726562500000, 0.00781250000000000, -0.00781250000000000, -0.0219726562500000, -0.0346679687500000, -0.0458984375000000, -0.0556640625000000, -0.0639648437500000, -0.0708007812500000, -0.0761718750000000, -0.0800781250000000, -0.0825195312500000, -0.0834960937500000, -0.0830078125000000, -0.0810546875000000, -0.0776367187500000, -0.0727539062500000, -0.0664062500000000, -0.0585937500000000, -0.0512695312500000, -0.0444335937500000, -0.0380859375000000, -0.0322265625000000, -0.0268554687500000, -0.0219726562500000, -0.0175781250000000, -0.0136718750000000, -0.0102539062500000, -0.00732421875000000, -0.00488281250000000, -0.00292968750000000, -0.00146484375000000, -0.000488281250000000, ], 
                    [6.10351562500000e-05, 0.000183105468750000, 0.000366210937500000, 0.000610351562500000, 0.000915527343750000, 0.00128173828125000, 0.00170898437500000, 0.00219726562500000, 0.00274658203125000, 0.00335693359375000, 0.00402832031250000, 0.00476074218750000, 0.00555419921875000, 0.00640869140625000, 0.00732421875000000, 0.00830078125000000, 0.00933837890625000, 0.0104370117187500, 0.0115966796875000, 0.0128173828125000, 0.0140991210937500, 0.0154418945312500, 0.0168457031250000, 0.0183105468750000, 0.0198364257812500, 0.0214233398437500, 0.0230712890625000, 0.0247802734375000, 0.0265502929687500, 0.0283813476562500, 0.0302734375000000, 0.0322265625000000, 0.0339965820312500, 0.0355834960937500, 0.0369873046875000, 0.0382080078125000, 0.0392456054687500, 0.0401000976562500, 0.0407714843750000, 0.0412597656250000, 0.0415649414062500, 0.0416870117187500, 0.0416259765625000, 0.0413818359375000, 0.0409545898437500, 0.0403442382812500, 0.0395507812500000, 0.0385742187500000, 0.0374145507812500, 0.0360717773437500, 0.0345458984375000, 0.0328369140625000, 0.0309448242187500, 0.0288696289062500, 0.0266113281250000, 0.0241699218750000, 0.0215454101562500, 0.0187377929687500, 0.0157470703125000, 0.0125732421875000, 0.00921630859375000, 0.00567626953125000, 0.00195312500000000, -0.00195312500000000, -0.00567626953125000, -0.00921630859375000, -0.0125732421875000, -0.0157470703125000, -0.0187377929687500, -0.0215454101562500, -0.0241699218750000, -0.0266113281250000, -0.0288696289062500, -0.0309448242187500, -0.0328369140625000, -0.0345458984375000, -0.0360717773437500, -0.0374145507812500, -0.0385742187500000, -0.0395507812500000, -0.0403442382812500, -0.0409545898437500, -0.0413818359375000, -0.0416259765625000, -0.0416870117187500, -0.0415649414062500, -0.0412597656250000, -0.0407714843750000, -0.0401000976562500, -0.0392456054687500, -0.0382080078125000, -0.0369873046875000, -0.0355834960937500, -0.0339965820312500, -0.0322265625000000, -0.0302734375000000, -0.0283813476562500, -0.0265502929687500, -0.0247802734375000, -0.0230712890625000, -0.0214233398437500, -0.0198364257812500, -0.0183105468750000, -0.0168457031250000, -0.0154418945312500, -0.0140991210937500, -0.0128173828125000, -0.0115966796875000, -0.0104370117187500, -0.00933837890625000, -0.00830078125000000, -0.00732421875000000, -0.00640869140625000, -0.00555419921875000, -0.00476074218750000, -0.00402832031250000, -0.00335693359375000, -0.00274658203125000, -0.00219726562500000, -0.00170898437500000, -0.00128173828125000, -0.000915527343750000, -0.000610351562500000, -0.000366210937500000, -0.000183105468750000, -6.10351562500000e-05, ], 
                    ]
    
    Grid_size = 1024    
    # Creo una grilla de muestreo en frecuencia logaritmica para que la
    # zona en que derivan los filtros sea una recta de pendiente constante.
    F_log = logspace(-3, min([0, log10(fs/f_ref)]), Grid_size) * pi
    
    for ii in scales:
    
        F, g_amp = freqz(empirical_tf[ii],1, F_log)
        F = F * f_ratio / pi
        g_amp = 20*log10(abs(g_amp)+eps)
    
        #averiguo hasta qu� muestra se comporta como un derivador, ya que ser�
        #un par�metro de dise�o.
        slope_aux = diff( g_amp )
        
        end_diff_idx = find_idx(slope_aux < 0.9*slope_aux[0])[0]

#        plt.plot( slope_aux[:end_diff_idx])
#        plt.show()
        
        #dise�o un derivador hasta dicha frecuencia, con una banda de
        #transici�n dada por la expresion , cuando sea posible.
        ftrans = min([70, 251 * exp(-0.63*(ii+1))])
        diff_order = round(N * 1.8**(((ii+1)*0.4 - 0.6)) ) 
        #Los N tienen que ser necesariamente pares para el dise�o del derivador.
        if( diff_order % 2 != 0 ):
            diff_order += 1


        # comienzo de la banda de atenuación, cuando alcanza el 95% de la atenuación 
        # en Nyquist. Evito Nyq dado que a veces hay -inf        
        att_stopband = mean(g_amp[ int(mean([end_diff_idx, Grid_size])):-2 ])
        start_stop_idx = find_idx(g_amp < 0.95*att_stopband)[0] 
        
        if(start_stop_idx <= end_diff_idx): 
            # no decae nunca el derivador, fijo una transición relajada
            start_stop_idx = int(end_diff_idx + 0.9*(Grid_size-end_diff_idx))
        
        
        #[msgstr, msgid] = lastwarn
        #itero hasta que se dise�a correctamente.
        jj = 0
        effective_order = diff_order
        aux_seq = linspace(0.5, 1.1, design_iter_attemps)

        fstop = min( [0.95, F[end_diff_idx]+(ftrans*2/fs)])
        fstop_idx = find_idx( F >= fstop )[0]
        
        while( jj < design_iter_attemps ):
            
            try:
                
                designed_ok = False;
                Hd = remez( int(effective_order), [0.0, F[end_diff_idx], F[start_stop_idx], 1.0], [1.0, 0.0], type='differentiator', fs = 2 )
                
            except ValueError as ex_ve:
                
#                template = "Arguments:{0!r}"
#                message = template.format(ex_ve.args)
#                print(message)
                #fallo en la convergencia, iteramos de nuevo.
                #recorro linealmente por el rango +10:-50
                effective_order = round(diff_order * aux_seq[jj])
                #Los effective_order tienen que ser necesariamente pares para el dise�o del derivador.
                if( effective_order % 2 != 0 ):
                    effective_order += 1
                
                # debug            
                if debug:
                    print( 'Trying order {0}\n'.format(effective_order))

                jj += 1
                
            except Exception as ex:
                
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)
                
            else:
                designed_ok = True;
                jj = design_iter_attemps
            
        if(not designed_ok):
            #fallo en la convergencia, reportamos el error.
#            ('qs_filter_design:Impossible2Design', '')
            raise( ValueError("Impossible to design the differentiator filter. Please try another N value, or check the filters transfer functions manually.") )

        
        #Vuelvo a ver la transferencia real del derivador dise�ado y del filtro
        #a emular para que tengan una respuesta similar en la zona en que se
        #comporta como derivador. Averiguo el factor de escala entre ambas
        #transferencias.
        
        _, g_amp_real = freqz(Hd, 1, F * pi)
        g_amp_real = 20*log10(abs(g_amp_real))
        
        aux_idx = int(round(end_diff_idx/2))
        aux_scale = g_amp[aux_idx]-g_amp_real[aux_idx]
        
        #Escalo la zona del derivador.
        Hd = Hd * 10**(aux_scale/20)

        # debug: show frequency response            
        if debug:
            F_ref = logspace(-3, log10(f_ref/2), Grid_size)
            F_tgt = logspace(-3, log10(fs/2), Grid_size)

            _,g_amp_real = freqz(Hd, 1, F_tgt, fs = fs ); g_amp_real = 20*log10(abs(g_amp_real)); 
            _,g_amp_ref = freqz(empirical_tf[ii], 1, F_ref, fs = f_ref ); g_amp_ref = 20*log10(abs(g_amp_ref)); 
            
            plt.plot(F_tgt, g_amp_real, label='designed@'+str(fs)+'Hz'); 
            plt.plot(F_ref, g_amp_ref, label='paper@250Hz'); 
            plt.axvspan(0, F[end_diff_idx], color='red', alpha=0.2, label='diff. BW'); 
            plt.legend(); 
            
            plt.ylim([1.1*min(g_amp_real), 1.1*max([g_amp_real, g_amp_ref]) ]); 
            plt.title('Scale ' + str(ii+1) + ': full bandwith'); 
            plt.show()

            F_diff = logspace(-3, log10(f_ref/2), Grid_size)

            _,g_amp_real = freqz(Hd, 1, F_diff, fs = fs ); g_amp_real = abs(g_amp_real); 
            _,g_amp_ref = freqz(empirical_tf[ii], 1, F_diff, fs = f_ref ); g_amp_ref = abs(g_amp_ref); 
            
            plt.cla()
            plt.loglog(F_diff, g_amp_real, label='designed@'+str(fs)+'Hz'); 
            plt.loglog(F_diff, g_amp_ref, label='paper@250Hz'); 
            plt.axvspan(0, F[end_diff_idx], color='red', alpha=0.2, label='diff. BW'); 
            plt.legend(); 
            
            plt.ylim([1.1*min(g_amp_real), 1.1*max([g_amp_real, g_amp_ref]) ]); 
            plt.title('Escala ' + str(ii+1) + ': differentiator bandwidth'); 
            plt.show()
    
        q_filters += [Hd]
        
        filter_count += 1
    
    return(q_filters)
    
if __name__ == '__main__':
#    qs_filter_design( )
    qs_filter_design( scales = np.arange(1,5), fs = 360 )
    

    