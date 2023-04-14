from __future__ import division
import numpy as np
import pandas as pd
import time
import os
import random
import argparse
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Input, Dropout, Dense, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from graph_attention_layer import GraphAttention
from soft_clustering_layer import ClusteringLayer
from utils import load_data, my_kmeans, saveClusterResult

my_seed = 42
np.random.seed(my_seed)
random.seed(my_seed)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_str', default='Biase', type=str, help='name of dataset')
parser.add_argument('--n_clusters', default=3, type=int, help='expected number of clusters')
parser.add_argument('--label_path', default='data/Biase/label.ann', type=str, help='true labels')
parser.add_argument('--c0', default=1, type=float, help='weight of AE reconstruction loss')
parser.add_argument('--c1', default=1, type=float, help='weight of GAT reconstruction loss')
parser.add_argument('--c2', default=1, type=float, help='weight of clustering loss')
parser.add_argument('--k', default=None, type=int, help='number of neighbors to construct the cell graph')
parser.add_argument('--is_NE', default=True, type=bool, help='use NE denoise the cell graph or not')
parser.add_argument('--PCA_dim', default=512, type=int,help='dimensionality of input feature matrix that transformed by PCA')
parser.add_argument('--F1', default=128, type=int, help='number of neurons in the 0-st layer of encoder')
parser.add_argument('--F2', default=64, type=int, help='number of neurons in the 1-st layer of encoder')
parser.add_argument('--F3', default=16, type=int, help='number of neurons in the 2-nd layer of encoder')
parser.add_argument('--n_attn_heads', default=4, type=int, help='number of heads for attention')
parser.add_argument('--dropout_rate', default=0.4, type=float, help='dropout rate of neurons in autoencoder')
parser.add_argument('--l2_reg', default=0, type=float, help='coefficient for L2 regularizition')
parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate for training')
parser.add_argument('--pre_lr', default=2e-4, type=float, help='learning rate for pre-training')
parser.add_argument('--pre_epochs', default=200, type=int, help='number of epochs for pre-training')
parser.add_argument('--epochs', default=5000, type=int, help='number of epochs for training')
args = parser.parse_args()

if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists('result/'):
    os.makedirs('result/')

dataset_str = args.dataset_str
n_clusters = args.n_clusters
dropout_rate = args.dropout_rate

# Paths
data_path = 'data/' + dataset_str + '/data.tsv'
DF_autoencoder_path = 'logs/scDFC_' + dataset_str + '.h5'
model_path = 'logs/model_' + dataset_str + '.h5'
pred_path = 'result/pred_' + dataset_str + '.txt'
intermediate_path = 'logs/model_' + dataset_str + '_'

# Read data
start_time = time.time()
A, X, cells, genes = load_data(data_path, dataset_str,
                               args.PCA_dim, args.is_NE, n_clusters, args.k)

end_time = time.time()
run_time = (end_time - start_time) / 60
print('Pre-process: run time is %.2f ' % run_time, 'minutes')

# Parameters
N = X.shape[0]  # Number of nodes
F = X.shape[1]  # Original feature dimension

# Fusion Model definition
Xa_in = Input(shape=(F,))
X_in = Input(shape=(F,))
A_in = Input(shape=(N,))
# encoder
# AE
AE_dropout1 = Dropout(dropout_rate)(Xa_in)
AE_encode_1 = Dense(512, activation="relu")(AE_dropout1)

AE_dropout2 = Dropout(dropout_rate)(AE_encode_1)
AE_encode_2 = Dense(256, activation="relu")(AE_dropout2)

AE_dropout3 = Dropout(dropout_rate)(AE_encode_2)
AE_encode_3 = Dense(64, activation="relu")(AE_dropout3)

dropout1 = Dropout(dropout_rate)(X_in)
graph_attention_1 = GraphAttention(args.F1,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout1, A_in])

dropout2 = Dropout(dropout_rate)(graph_attention_1)
graph_attention_2 = GraphAttention(args.F2,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout2, A_in])

dropout3 = Dropout(dropout_rate)(graph_attention_2)
graph_attention_3 = GraphAttention(args.F3,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout3, A_in])

Concat_1 = concatenate([AE_encode_3, graph_attention_3], axis=1)
Concat_dropout1 = Dropout(dropout_rate)(Concat_1)

Concat_2 = Dense(64, activation="relu")(Concat_dropout1)
Concat_dropout2 = Dropout(dropout_rate)(Concat_2)

AE_decode_1 = Dense(256, activation="relu")(Concat_dropout2)

AE_dropout4 = Dropout(dropout_rate)(AE_decode_1)
AE_decode_2 = Dense(512, activation="relu")(AE_dropout4)

AE_dropout5 = Dropout(dropout_rate)(AE_decode_2)
AE_decode_3 = Dense(F, activation="relu")(AE_dropout5)

graph_attention_4 = GraphAttention(args.F2,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([Concat_dropout2, A_in])

dropout4 = Dropout(dropout_rate)(graph_attention_4)
graph_attention_5 = GraphAttention(args.F1,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout4, A_in])

dropout5 = Dropout(dropout_rate)(graph_attention_5)
graph_attention_6 = GraphAttention(128,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout5, A_in])

# Build DF autoencoder model
DF_autoencoder = Model(inputs=[X_in, A_in, Xa_in], outputs=[graph_attention_6, AE_decode_3])
optimizer = Adam(lr=args.pre_lr)
DF_autoencoder.compile(optimizer=optimizer,
                       loss=['cosine', 'mse'], loss_weights=[args.c0, args.c1])

# Callbacks
es_callback = EarlyStopping(monitor='loss', min_delta=0.05, patience=50)
tb_callback = TensorBoard(batch_size=N)
mc_callback = ModelCheckpoint(DF_autoencoder_path,
                              monitor='loss',
                              save_best_only=True,
                              save_weights_only=True)

# Train DF_autoencoder model
start_time = time.time()
DF_autoencoder.fit([X, A, X], [X, X], epochs=args.pre_epochs, batch_size=N, verbose=0, shuffle=False)
end_time = time.time()
run_time = (end_time - start_time) / 60
print('Pre-train: run time is %.2f ' % run_time, 'minutes')

# Construct a model for hidden layer
hidden_model = Model(inputs=DF_autoencoder.input, outputs=Concat_2)
hidden = hidden_model.predict([X, A, X], batch_size=N)
hidden = hidden.astype(float)

# Get k-means clustering results of hidden representation of cells
y_pred, pre_centers = my_kmeans(n_clusters, hidden, dataset_str)
y_pred_last = np.copy(y_pred)

# Add the soft_clustering layer
soft_cluster_layer = ClusteringLayer(n_clusters,
                                     N,
                                     'q',
                                     0,
                                     pre_centers,
                                     name='clustering')(Concat_dropout2)

def pred_loss(y_true, y_pred):
    return y_pred

# Construct total model
model = Model(inputs=[X_in, A_in, Xa_in],
              outputs=[graph_attention_6,
                       AE_decode_3,
                       soft_cluster_layer,
                       Concat_2])

optimizer = Adam(lr=args.learning_rate)
model.compile(optimizer=optimizer,
              loss=['cosine', 'mse', 'kld', pred_loss],
              loss_weights=[args.c0, args.c1, args.c2, 0])

# Train model
start_time = time.time()

tol = 1e-5
loss = 0

sil_logs = []
update_interval = 2
res_ite = 0
final_pred = None
max_sil = 0

for ite in range(args.epochs + 1):
    if ite % update_interval == 0:
        res_ite = ite

        _, _, q, hid = model.predict([X, A, X], batch_size=N, verbose=0)
        p = q ** 2 / q.sum(0)
        p = (p.T / p.sum(1)).T
        y_pred = q.argmax(1)

        sil_hid = metrics.silhouette_score(hid, y_pred, metric='euclidean')
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        print('Iter:', ite,
              ', sil_hid:', np.round(sil_hid, 3),
              ', delta_label', np.round(delta_label, 3),
              ', loss:', np.round(loss, 2))

        sil_logs.append(sil_hid)
        arr_sil = np.array(sil_logs)

        if sil_hid >= max_sil:
            final_pred = y_pred
            max_sil = sil_hid

        if len(arr_sil) >= 30 * 2:
            mean_0_n = np.mean(arr_sil[-30:])
            mean_n_2n = np.mean(arr_sil[-60: -30])
            if mean_0_n - mean_n_2n <= 0.02:
                print('Stop early at', ite, 'epoch')
                break

        if len(arr_sil) >= 3:
            if arr_sil[-2] - arr_sil[-1] >= 0.05:
                print('Stop early at', ite, 'epoch')
                break

    loss = model.train_on_batch(x=[X, A, X], y=[X, X, p, hid])

# model.save_weights(model_path)

end_time = time.time()
run_time = (end_time - start_time)
print('Train: run time is %.2f ' % run_time, 'seconds')

saveClusterResult(final_pred, cells, dataset_str)

if args.label_path:
    true_path = args.label_path
    true = pd.read_csv(true_path, sep='\t').values
    true = true[:, -1].astype(int)
    ARI = adjusted_rand_score(final_pred, true)
    NMI = metrics.normalized_mutual_info_score(true, y_pred)

    print('#######################')
    print('ARI {}'.format(ARI))
    print('NMI {}'.format(NMI))

# Get hidden representation
hidden_model = Model(inputs=model.input, outputs=graph_attention_2)
hidden = hidden_model.predict([X, A, X], batch_size=N)
hidden = hidden.astype(float)

mid_str = dataset_str
hidden = pd.DataFrame(hidden)
hidden.to_csv('result/hidden_' + mid_str + '.tsv', sep='\t')
print('Done.')
