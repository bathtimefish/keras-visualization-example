from keras.models import model_from_json

model = None
with open('model.json') as f:
    model = model_from_json(f.read())

model.load_weights('weights.hdf5')

layer_num = 0   # 0=畳み込み1層目
print('Layer Name: {}'.format(model.layers[layer_num].get_config()['name']))
W = model.layers[layer_num].get_weights()[0]

W = W.transpose(3, 2, 0, 1) # 配列を転置する
nb_filter, nb_channel, nb_row, nb_col = W.shape

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
plt.figure()
for i in range(nb_filter):
    im = W[i, 0]
    # 重みを0-255のスケールに変換する
    scaler = MinMaxScaler(feature_range=(0, 255))
    im = scaler.fit_transform(im)
    # プロットを1つの図にまとめる 4行x8列=32個
    plt.subplot(4, 8, i + 1)
    plt.axis('off')
    plt.imshow(im, cmap='gray')
plt.show()

