from data import *
from mergedgenerators import MergedGenerators
from model import *

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.visible_device_list = '0'
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.compat.v1.Session(config = config)

from keras import backend as K
K.set_session(sess)

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

trainlist = []
trainannotlist = []
for i in range(1,6):
    trainlist.append('Fold'+str(i))
    trainlist.append('Fold'+str(i)+'_pre')
    trainannotlist.append('Fold'+str(i)+'_label')
    trainannotlist.append('Fold'+str(i)+'_pre_label')
    
myGenerator = trainGenerator(32,'ETT_TRAIN',trainlist,trainannotlist,
                        data_gen_args)#,save_to_dir = 'ETT_v3/traingenerator'


model = unet()
model_checkpoint = ModelCheckpoint('unet_ETT_merge.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGenerator,steps_per_epoch=250,epochs=50,callbacks=[model_checkpoint])
