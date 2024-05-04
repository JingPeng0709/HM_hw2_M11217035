from data import *
from mergedgenerators import MergedGenerators
from model import *

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

model = unet()
model.load_weights("unet_ETT_fully.hdf5")

testGene = testGenerator("ETT_TEST/Fold1/test")
#testGene = testGenerator("ETT_TRAIN/Fold1")

results = model.predict_generator(testGene,30,verbose=1)
#saveResult("ETT_TEST/Fold1/result",results)
saveResult("ETT_TEST/Fold1/result",results)