project_config = {
    'destination_ip': {'conext_data': '192.168.1.107', 'Data_combined': '192.168.1.187', '13May': 'dynamic', 'IMC_Lab_data': '192.168.1.187', 'th': '192.168.1.187', 'th_jitter': '192.168.1.187', 'lat': '192.168.1.187', 'lat_jitter': '192.168.1.187', 'loss': '192.168.1.187'},
    'webrtc_anomaly_threshold': 0.5,
    'prediction_window': 1,
    'n_features_size': {'teams': 800, 'meet': 600, 'webex': 600},
    'n_features_bps': {'teams': 800, 'meet': 600, 'webex': 600},
    'n_features_iat': {'teams': 800, 'meet': 600, 'webex': 600},
    'video_thresh': 306,
    'webrtc_format': {'conext_data': 1, 'Data_combined': 2, '13May': 2, 'IMC_Lab_data': 2, 'th': 2, 'th_jitter': 2, 'lat': 2, 'lat_jitter': 2, 'loss': 2},
    'data_format': {'conext_data': 2, 'Data_combined': 3, '13May': 4, 'IMC_Lab_data': 1, 'th': 1, 'th_jitter': 1, 'lat': 1, 'lat_jitter': 1, 'loss': 1},
    'jitter_buffer_size': {'teams': 4, 'meet': 0.025, 'webex': 0.01},
    'features_list': {
        'ml': ['et', 'l_max', 'l_mean', 'l_min', 'l_num_bytes', 'l_num_pkts',
       'l_num_unique', 'l_q2', 'l_std', 't_max',
       't_mean', 't_min', 't_q2', 't_std', 't_burst_count'],
        'rtp_ml': ['et', 'l_max', 'l_mean', 'l_min', 'l_num_bytes', 'l_num_pkts', 'l_q2', 'l_std', 't_max',
       't_mean', 't_min', 't_q2', 't_std','vid_ts_unique', 'rtx_ts_unique', 'vid_marker_sum', 'rtx_marker_sum', 'ooo_seqno_vid', 'rtp_lag_min','rtp_lag_max','rtp_lag_q2','rtp_lag_mean','rtp_lag_std']
    },
    'rtx_ptype' : {'conext_data': {'meet': ['99'], 'teams': ['121']}, 'Data_combined': {'meet': ['99'], 'teams': ['123'], 'webex': [None]}, '13May': {'meet': ['99'], 'teams': ['101'], 'webex': [None]}, 'IMC_Lab_data': {'meet': ['99'], 'webex': [None], 'teams': ['103']}, 'th': {'meet': ['99'], 'webex': [None], 'teams': ['103']},'th_jitter': {'meet': ['99'], 'webex': [None], 'teams': ['103']},'lat': {'meet': ['99'], 'webex': [None], 'teams': ['103']},'lat_jitter': {'meet': ['99'], 'webex': [None], 'teams': ['103']},'loss': {'meet': ['99'], 'webex': [None], 'teams': ['103']}},
    'video_ptype': {'conext_data': {'meet': ['96', '98'], 'teams': ['127']}, 'Data_combined': {'meet': ['98'], 'teams': ['102'], 'webex': ['102']}, '13May': {'meet': ['98'], 'teams': ['100'], 'webex': ['100']}, 'IMC_Lab_data': {'meet': ['96', '98'], 'webex': ['102'], 'teams': ['102']}, 'th': {'meet': ['96', '98'], 'webex': ['102'], 'teams': ['102']}, 'th_jitter': {'meet': ['96', '98'], 'webex': ['102'], 'teams': ['102']},'lat': {'meet': ['96', '98'], 'webex': ['102'], 'teams': ['102']},'lat_jitter': {'meet': ['96', '98'], 'webex': ['102'], 'teams': ['102']},'loss': {'meet': ['96', '98'], 'webex': ['102'], 'teams': ['102']}}
}
