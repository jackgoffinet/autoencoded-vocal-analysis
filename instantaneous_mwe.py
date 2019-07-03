"""
# -------------------------------------- #
#        Instantaneous Song Stuff        #
# -------------------------------------- #
from preprocessing.template_segmentation import process_sylls, clean_collected_data
syll_types = ["E", "A", "B", "C", "D", "call"]
# temp = ['09262019/', '09292019/', 'IMAGING_09182018/', 'IMAGING_09242018/', \
# 		'IMAGING_10112018/', 'IMAGING_10162018/']
temp = ['09262019/']
load_dirs = ['data/raw/bird_data/' + i for i in temp]
temp_save_dirs = ['data/processed/bird_data/temp_' + i for i in temp]
save_dirs = ['data/processed/bird_data/' + i for i in temp]
feature_dirs = ['data/features/blk215/' + i for i in syll_types]

p = {
	'songs_per_file': 20,
	'num_freq_bins': 128,
	'num_time_bins': 128,
	'min_freq': 350,
	'max_freq': 10e3,
	'mel': True,
	'spec_thresh': 1.0,
}

# Preprocessing
for syll_type, feature_dir in zip(syll_types, feature_dirs):
	print("Syllable:", syll_type)
	temp_save_dirs = ['data/processed/bird_data/temp_' + syll_type + '_'+ i for i in temp]
	save_dirs = ['data/processed/bird_data/' + syll_type + '_' + i for i in temp]
	for load_dir, temp_save_dir in zip(load_dirs, temp_save_dirs):
		process_sylls(load_dir, temp_save_dir, feature_dir, p)
	clean_collected_data(temp_save_dirs, save_dirs, p)

quit()
# Training: syllables
from models.dlgm import DLGM
from models.dataset import get_partition, get_data_loaders
partition = get_partition(save_dirs, split=0.95)
# Check load_dir vs. save_dir!
model = DLGM(network_dims, partition=partition, save_dir='data/models/red215_syll/', sylls_per_file=p['songs_per_file'])
model.train(epochs=250, lr=2e-5)
quit()

# Training: fixed window
from models.fixed_window_dlgm import DLGM
from models.fixed_window_dataset import get_partition, get_data_loaders
partition = get_partition(save_dirs, split=0.95)
# Check load_dir vs. save_dir!
model = DLGM(network_dims, p, partition=partition, load_dir='data/models/blk215_inst/')
# model.train(epochs=250, lr=1.5e-5)

partition = get_partition(['data/processed/bird_data/IMAGING_09182018/'], split=1.0)
# temp = ['IMAGING_09242018/', 'IMAGING_10112018/', \
		# 'IMAGING_10162018/'] # '09262019/', '09292019/',
# temp_dirs = ['data/processed/bird_data/'+i for i in temp]
# partition = get_partition(temp_dirs, split=1.0)
loader, _ = get_data_loaders(partition, p, shuffle=(False, False))


# n = min(len(loader.dataset), 400)
# latent_paths = np.zeros((n,200,latent_dim))
# print("loader:", len(loader.dataset))
# from tqdm import tqdm
# for i in tqdm(range(n)):
# 	latent, ts = model.get_song_latent(loader, i, n=200)
# 	latent_paths[i] = latent
# np.save('latent_paths_other.npy', latent_paths)
# quit()

ts = np.linspace(0,0.85,200)
# latent_paths_1 = np.load('latent_paths_imaging.npy')
# latent_paths_2 = np.load('latent_paths_other.npy')
# latent_paths = np.concatenate((latent_paths_1, latent_paths_2), axis=0)

latent_paths = np.load('latent_paths_imaging.npy')

from plotting.instantaneous_plots import plot_paths_imaging
for unit in range(53):
	plot_paths_imaging(np.copy(latent_paths), ts, loader, unit_num=unit, filename=str(unit).zfill(2)+'.pdf')
quit()
# -------------------------------------- #
"""
