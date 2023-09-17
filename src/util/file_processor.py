import os
from .pcap2csv import convert
from glob import glob
from config import project_config
from .validator import FileValidator


class FileProcessor:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def get_linked_files(self):
        bname = os.path.basename(self.data_directory)
        if bname == 'in_lab_data':
            return self.get_in_lab_files()
        elif bname == 'hashed_in_lab':
            return self.get_in_lab_files()
        elif bname == 'real_world_data':
            return self.get_real_world_files()
        elif bname == 'hashed_real_world':
            return self.get_real_world_files()
        return dict()

    def get_in_lab_files(self):
        linked_files = {}
        for experiment_dir in os.listdir(self.data_directory):
            vca = experiment_dir.split('_')[1]
            experiment_dir = os.path.join(self.data_directory, experiment_dir)
            if vca not in linked_files:
                linked_files[vca] = []
            try:
                csv_file = glob(f'{experiment_dir}/*.csv')[0]
                webrtc_file = glob(f'{experiment_dir}/*.json')[0]
            except:
                print('Missing file. Discarding the experiment')

            file_tuple = (csv_file, webrtc_file)
            linked_files[vca].append(file_tuple)
        return linked_files

    def get_real_world_files(self):
        linked_files = {}
        for device in os.listdir(self.data_directory):
            dpath = f'{self.data_directory}/{device}'
            if os.path.exists(dpath):
                # convert(f'{dpath}')
                csvs = glob(f'{dpath}/*.csv')
                for csv in csvs:
                    vca = os.path.basename(csv).split('-')[1]
                    if vca not in linked_files:
                        linked_files[vca] = []
                    print(csv)
                    bname = os.path.basename(csv)
                    webrtc_file = bname[:-4] + ".json"
                    if webrtc_file not in os.listdir(f'{dpath}'):
                        print(webrtc_file)
                        print('WebRTC file not found')
                        continue
                    webrtc_file = f'{dpath}/{webrtc_file}'
                    file_tuple = (csv, webrtc_file)
                    linked_files[vca].append(file_tuple)
        return linked_files