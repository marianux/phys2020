function detection_script( examples_path, bUpdate, rec_idx )
%CONVERSION_SCRIPT Summary of this function goes here
%   Detailed explanation goes here

    if(nargin < 3 || isempty(rec_idx) )
        rec_idx = 1;
    end

    if(nargin < 2 || isempty(bUpdate) )
        bUpdate = false;
    end

    
    
    dirnames = dir([ examples_path filesep '*.hea'] );
    
    root_path = fileparts(mfilename('fullpath'));
    
    output_path = [ root_path filesep 'ecg_kit_tmp_files' filesep ];
    result_path = [ root_path filesep 'qrs_detections' filesep ];
    
    if( ~exist(result_path, 'dir') )
        mkdir(result_path)
    end

    
    % Arbitrary Impulsive Pseudoperiodic (AIP) configuration:
    % This is an unpublished (at the moment of writing this) detector for
    % arbitrary patterns, in this case, we briefly describe the pattern to
    % match in terms of:
    payload_in.trgt_width = 0.06; % 60 milliseconds width. A narrow and normal QRS complex
    % The minimum separation among heartbeats, 300 millisecond allow most of
    % the anticipated heartbeats to be detected. MITDB has even more
    % anticipated heartbeats, but 300 milliseconds is a decent starting point.
    payload_in.trgt_min_pattern_separation = 0.3; % seconds. 
    % The longest pause allowed among patterns. By the moment this property is
    % unused, but future releases can detect pauses in order to restart the
    % pattern search, or relax certain restrictions.
    payload_in.trgt_max_pattern_separation = 2; % seconds
    % amount of patterns to find, matching the above criteria. Could be related
    % to the heartbeats morphologies in QRS detection context.
    payload_in.max_patterns_found = 2; % patterns
    
    
    rec_found = length(dirnames);
    
    if( rec_found <= 0 )
        
        fprintf(2, 'No recs found in %s\n', examples_path);
    
    else

        fprintf(1, 'Found in %d recordings.\n', rec_found);
        
        for ii = rec_idx:rec_found

            fprintf(1, '%d %%  %d - %d %s\n', round(ii/rec_found*100), ii, rec_found, dirnames(ii).name );

            %% pECG preprocessing

            ECGw = ECGwrapper( 'recording_name', [ examples_path filesep dirnames(ii).name ] );

            ECGw.output_path = output_path;
            ECGw.ECGtaskHandle = 'QRS_detection';
            ECGw.ECGtaskHandle.only_ECG_leads = false;
            ECGw.ECGtaskHandle.detectors = { 'wavedet', 'gqrs'};

            ECGw.cacheResults = ~bUpdate; 

            ECGw.Run

            cached_filenames = ECGw.Result_files;
            QRS_detectors_struct = load(cached_filenames{1});

            % Arbitrary Impulsive Pseudoperiodic (AIP) detector: Another
            % unpublished detector

            ECGw.ECGtaskHandle = 'arbitrary_function';

            ECGw.ECGtaskHandle.payload = payload_in;

            % Add a user-string to identify the run
            ECGw.user_string = 'AIP_det';

            % add your function pointer
            ECGw.ECGtaskHandle.function_pointer = @aip_detector;
            ECGw.ECGtaskHandle.concate_func_pointer = @aip_detector_concatenate;

            ECGw.cacheResults = ~bUpdate; 

            ECGw.Run

            cached_filenames = ECGw.Result_files;
            QRS_aip_struct = load(cached_filenames{1});

        %% Post process detections
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % Merge together detections generated with:
            % 
            %   + Wavedet
            %   + GQRS
            %   + AIP
            % 

            ECGw.ECGtaskHandle = 'QRS_detections_post_process';

            % Mix QRS detections strategy function
            % *mixartif* use the output of *all* leads and detectors to perform
            % a multi-lead-and-algorithm composition. The result of this
            % algorithm is a new set of detections based on the concatenation
            % of the ""best"" detections found for each 20-seconds window in a
            % recording. So, this algorithm generates *new* QRS detection
            % sieres.
            ECGw.ECGtaskHandle.post_proc_func = 'mixartif';

            % Add a user-string to identify the run
            ECGw.user_string = 'mixartif';

            QRS_detections = concat_QRS_detections( QRS_detectors_struct, QRS_aip_struct);

            ECGw.ECGtaskHandle.payload = QRS_detections;

            ECGw.cacheResults = ~bUpdate; 

            ECGw.Run

            cached_filenames = ECGw.Result_files;

            mixartif_struct = load(cached_filenames{1});

            % Lead selection strategy based on the best (quality metric) *m*.
            % This is a lead selection algorithm, so this algorithm filter the
            % first ranked lead of all the algorithms outputs included in the
            % payload (QRS_detections struct)
            ECGw.ECGtaskHandle.post_proc_func = 'best_m_lead';

            % Add a user-string to identify the run
            ECGw.user_string = 'best_m_lead';

            ECGw.ECGtaskHandle.payload = QRS_detections;

            ECGw.cacheResults = ~bUpdate; 

            ECGw.Run

            cached_filenames = ECGw.Result_files;

            best_m_struct = load(cached_filenames{1});

            QRS_detections = concat_QRS_detections( mixartif_struct, best_m_struct);

            save( [result_path filesep dirnames(ii).name(1:end-4) '_QRS_detections' ], '-struct', 'QRS_detections')

        end

    end

end

