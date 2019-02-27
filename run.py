from vgg16 import * 
from glob import glob
import matplotlib 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
from pylab import subplots 

# model
model = VGG16(include_top=True, weights='imagenet') 
print("İşlenen resim:", "static/fish.jpg")

# get the .jpg file in targets folder
phishies = glob("./targets/*.jpg") 

predictions = [] 

for fish in phishies: 

    # load image [[RED], [GREEN], [BLUE]]
    img = image.load_img(fish, target_size=(224, 224)) 
    x = image.img_to_array(img) 
    x = np.expand_dims(x, axis=0) # [233, 233, 123] -> [123123] -> poolling -> flattening
    x = preprocess_input(x) # BGR 
    print("Tahmin ediliyor...")

    # prediction
    preds = model.predict(x)    
    
   # get the highest prediction
    pred =  decode_predictions(preds)    
    predictions.append({
        "pred": pred[0][-1], 
        "real": fish.split("\\")[-1].split(".")[0],
        "img":  img
    })


# sample number
nsamples = len(phishies) 
ncolumns = 2 

# partition of the samples to the columns
fig, ax = subplots(nsamples//ncolumns, ncolumns, figsize=(50, 50)) 
matplotlib.rcParams['font.size'] = 9


plt.subplots_adjust(hspace=.4, wspace=.4)

total = 0 
for i, rows in enumerate(ax):   
    for j, cell  in enumerate(rows):
        nd = predictions[total] 
        cell.set_title("Prediction: {} Truth: {}".format(nd['pred'], nd['real']))
        cell.imshow(nd['img'])
        total += 1

# show
plt.show()
