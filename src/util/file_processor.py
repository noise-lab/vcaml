import os
from .pcap2csv import convert
from glob import glob
from config import project_config
from .validator import FileValidator
"""
    Reads CSV and WebRTC files from a data directory and returns a list of tuples containing linked CSV and JSON files for each VCA.
    Assumes a fixed directory structure.

    Input: data_directory (absolute path to project's root)
"""
class FileProcessor:
    def __init__(self, data_directory, data_format=1):
        self.data_directory = data_directory
        self.data_format = data_format

    def get_linked_files(self):
        if self.data_format == 1:
            return self.use_data_format_1()
        elif self.data_format == 2:
            return self.use_data_format_2()
        elif self.data_format == 3:
            return self.use_data_format_3()
        elif self.data_format == 4:
            return self.use_data_format_4()

    def use_data_format_1(self):
        linked_files = {}
        for experiment_dir in os.listdir(self.data_directory):
            experiment_dir = os.path.join(self.data_directory, experiment_dir)
            vca = os.listdir(experiment_dir)[0]
            if vca not in linked_files:
                linked_files[vca] = []
            csv_path = f'{experiment_dir}/{vca}/captures'
            webrtc_path = f'{experiment_dir}/{vca}/webrtc'

            convert(csv_path)

            pcap_files = [x for x in os.listdir(csv_path) if x.endswith('pcap')]
            if len(pcap_files) == 0:
                raise MissingFileException(f'CSV file not found for the experiment: {experiment_dir}')
            else:
                pcap_file = f'{csv_path}/{pcap_files[0]}'
                
            csv_files = [x for x in os.listdir(csv_path) if x.endswith('csv') and not ('_ml_' in x or '_rtp_ml_' in x)]
            if len(csv_files) == 0:
                raise MissingFileException(f'CSV file not found for the experiment: {experiment_dir}')
            else:
                csv_file = f'{csv_path}/{csv_files[0]}'
            webrtc_files = [x for x in os.listdir(webrtc_path) if x.endswith('json')]
            if len(webrtc_files) == 0:
                print('WebRTC file not found')
                continue
            else:
                webrtc_file = f'{webrtc_path}/{webrtc_files[0]}'
            file_tuple = (pcap_file, csv_file, webrtc_file)
            validator = FileValidator(file_tuple, project_config, os.path.basename(self.data_directory))
            if validator.validate():    
                linked_files[vca].append(file_tuple)
        return linked_files

    def use_data_format_2(self):
        linked_files = {}
        for experiment_dir in os.listdir(self.data_directory):
            experiment_dir = os.path.join(self.data_directory, experiment_dir)
            # convert(f'{experiment_dir}/capture')
            vca = os.path.basename(experiment_dir).split('_')[1].lower()
            linked_files[vca] = []
            for pcap_file in os.listdir(f'{experiment_dir}/capture'):
                if pcap_file.endswith('.pcap'):
                    csv_file = f'{experiment_dir}/capture/{pcap_file[:-5]}.csv'
                    pref = '_'.join((pcap_file.split('_')[:5]))
                    webrtc_file = [x for x in os.listdir(f'{experiment_dir}/webrtc') if pref in x][0]
                    webrtc_file = f'{experiment_dir}/webrtc/{webrtc_file}'
                    linked_files[vca].append((pcap_file, csv_file, webrtc_file))
        return linked_files

    def use_data_format_3(self):
        linked_files = {}
        for vca in os.listdir(self.data_directory+'/'+'vca_data'):
            for experiment_dir in os.listdir(self.data_directory+'/'+'vca_data'+'/'+vca+'/'+'upton_data'):
                if experiment_dir.startswith('markov'):
                    continue
                experiment_dir = os.path.join(self.data_directory+'/'+'vca_data'+'/'+vca+'/'+'upton_data', experiment_dir)
                if vca not in linked_files:
                    linked_files[vca] = []
                csv_path = f'{experiment_dir}/{vca}/captures'
                webrtc_path = f'{experiment_dir}/{vca}/webrtc'
                annotated_fps_path = f'{experiment_dir}/{vca}/rec'

                # convert(csv_path)

                trace_id = os.path.basename(experiment_dir).split('_')[-1]

                trace_file = [x for x in os.listdir(self.data_directory+'/'+'traces') if x.endswith(f'{trace_id}.csv') ][0]
                trace_file = self.data_directory+'/'+'traces'+'/'+trace_file

                timestamp_file = [x for x in os.listdir(self.data_directory+'/'+'vca_data'+'/'+vca+'/router_timestamps') if x.endswith(f'{trace_id}.csv.txt')]

                if len(timestamp_file) == 0:
                    print(f'Timestamp file not found for the experiment: {experiment_dir}')
                    continue
                else:
                    timestamp_file = self.data_directory+'/'+'vca_data'+'/'+vca+'/router_timestamps/'+timestamp_file[0]

                pcap_files = [x for x in os.listdir(csv_path) if x.endswith('pcap')]
                if len(pcap_files) == 0:
                    print(f'PCAP file not found for the experiment: {experiment_dir}')
                    continue
                else:
                    pcap_file = f'{csv_path}/{pcap_files[0]}'
                    
                csv_files = [x for x in os.listdir(csv_path) if x.endswith('csv') and not ('_ml_' in x or '_rtp_ml_' in x)]
                if len(csv_files) == 0:
                    print(f'CSV file not found for the experiment: {experiment_dir}')
                    continue
                else:
                    csv_file = f'{csv_path}/{csv_files[0]}'
                webrtc_files = [x for x in os.listdir(webrtc_path) if x.endswith('json')]
                if len(webrtc_files) == 0:
                    print(f'WebRTC file not found for the experiment: {experiment_dir}')
                    continue
                else:
                    webrtc_file = f'{webrtc_path}/{webrtc_files[0]}'
                """
                annotated_fps_files = [x for x in os.listdir(annotated_fps_path) if x.endswith('.csv')]
                if 'fps.csv' not in annotated_fps_files or 'frame_level_info.csv' not in annotated_fps_files:
                    print(f'Annotated FPS file not found for the experiment: {experiment_dir}')
                    continue
                else:
                    fps_file = f'{annotated_fps_path}/fps.csv'
                    frame_file = f'{annotated_fps_path}/frame_level_info.csv'
                """
                linked_files[vca].append((pcap_file, csv_file, webrtc_file, trace_file, timestamp_file))
        return linked_files
    
    def use_data_format_4(self):
        linked_files = {}
        for device in  os.listdir(self.data_directory):
            dpath = f'{self.data_directory}/{device}/extract/tmp/Data'
            if os.path.exists(dpath):
                for vca in os.listdir(dpath):
                    # convert(f'{dpath}/{vca}')
                    if vca not in linked_files:
                        linked_files[vca] = []
                    pcaps = glob(f'{dpath}/{vca}/*.pcap')
                    for pcap in pcaps:
                        print(pcap)
                        bname = os.path.basename(pcap)
                        webrtc_file = bname[:-5] + ".json"
                        csv_file = bname[:-5] + ".csv"
                        log_file = bname[:-5] + ".log"
                        if webrtc_file not in os.listdir(f'{dpath}/{vca}'):
                            print(webrtc_file)
                            print('WebRTC file not found')
                            continue
                        if log_file not in os.listdir(f'{dpath}/{vca}'):
                            print(log_file)
                            print('Log file not found')
                            continue
                        webrtc_file = f'{dpath}/{vca}/{webrtc_file}'
                        csv_file = f'{dpath}/{vca}/{csv_file}'
                        log_file = f'{dpath}/{vca}/{log_file}'
                        file_tuple = (pcap, csv_file, webrtc_file, log_file, device)
                        validator = FileValidator(file_tuple, project_config, os.path.basename(self.data_directory))
                        if validator.validate():    
                            linked_files[vca].append(file_tuple)
        return linked_files

    def get_file_tuple(self, experiment_dir):
        vca = os.listdir(experiment_dir)[0]
        csv_path = f'{experiment_dir}/{vca}/captures'
        webrtc_path = f'{experiment_dir}/{vca}/webrtc'
        pcap_files = [x for x in os.listdir(csv_path) if x.endswith('pcap')]
        if len(pcap_files) == 0:
            raise MissingFileException(f'CSV file not found for the experiment: {experiment_dir}')
        else:
            pcap_file = f'{csv_path}/{pcap_files[0]}'
            
        csv_files = [x for x in os.listdir(csv_path) if x.endswith('csv')]
        if len(csv_files) == 0:
            raise MissingFileException(f'CSV file not found for the experiment: {experiment_dir}')
        else:
            csv_file = f'{csv_path}/{csv_files[0]}'
        webrtc_files = [x for x in os.listdir(webrtc_path) if x.endswith('json')]
        if len(webrtc_files) == 0:
            raise MissingFileException(f'WebRTC file not found for the experiment: {experiment_dir}')
        else:
            webrtc_file = f'{webrtc_path}/{webrtc_files[0]}'

        return (pcap_file, csv_file, webrtc_file)
        
class MissingFileException(Exception):
    pass
