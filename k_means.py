#encoding: utf-8
"""
@project=K_Means
@file=k_means
@author="QHX"
@create_time=2018/5/28 19:40
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
from mpl_toolkits.mplot3d import Axes3D
import copy
from matplotlib import colors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from segmenter import Transformer
from dr import DR
from lle import LLE
from hopkins import hopkins

def load_data(K):
    '''
    data_path="C://Users//15151//Desktop//xclara.csv"
    df=pd.read_csv(data_path)
    fig = plt.figure()
    sns.FacetGrid(data=df).map(plt.scatter,"V1","V2")
    plt.show()
    return df.values
    '''
    t = Transformer()
    t.segmenter()
    raw_data, name_list = t.numerizer(0.025)
    print(hopkins(copy.deepcopy(raw_data)))
    d = DR(raw_data, K)
    preprocessed_data = d.analyze()
    print(hopkins(copy.deepcopy(preprocessed_data)))
    return preprocessed_data, name_list

def data_pre_processing(data):
    scaler=StandardScaler()
    data_=scaler.fit_transform(data)
    return data_

def random_choose_K(data,K):
    random_number=5
    k_clusters=[]
    while K != 0:
        index = np.random.choice(data.shape[0], random_number, replace=False)
        if np.array(np.mean(data[index,:])) not in np.array(k_clusters):
            k_clusters.append(np.mean(data[index,:], axis=0))
            K -= 1
    '''
    for i in range(K):
        index=np.random.choice(data.shape[0],random_number,replace=False)
        k_clusters.append(np.mean(data[index,:],axis=0))
    '''
    print(k_clusters)
    return np.array(k_clusters)


def run_k_means(data,K):
    change=True
    k_clusters=random_choose_K(data,K)
    dist=np.zeros((data.shape[0],K))
    record_index=np.zeros(data.shape[0])
    iter=0
    while(change):
        iter+=1
        X_square=np.sum(data*data,axis=1)
        Y_square=np.sum(k_clusters*k_clusters,axis=1)
        X_Y_two=0-2*np.dot(data,k_clusters.T)
        dist=((X_square.reshape(-1,1)+X_Y_two).T+Y_square.reshape(-1,1)).T
        record_index_=np.argmin(dist,axis=1)
        for i in range(K):
            k_clusters[i]=np.mean(data[record_index_==i,:],axis=0)

        if np.sum(record_index==record_index_)==data.shape[0]:
            change=False
            return record_index_
        elif iter>100000:
            return record_index_
        else:
            pass


def visualize(src_2d, src_3d, label, names, K):
    # adjust font
    font_path = "/System/Library/Fonts/Hiragino Sans GB.ttc"
    prop = mfm.FontProperties(fname=font_path)
    # construct the map from cluster indexes to different colors
    table = dict(zip(range(K), colors.BASE_COLORS))
    # drawing
    plt.figure(1)
    plt.subplot(211)
    for index, cluster_label in enumerate(label):
        plt.scatter(src_2d[index][0], src_2d[index][1], c=table[cluster_label], alpha=0.5)
    plt.title("d=2")
    ax = plt.figure().add_subplot(212, projection='3d')
    for index, cluster_label in enumerate(label):
        ax.scatter(src_3d[index][0].real, src_3d[index][1].real, src_3d[index][2].real, c=table[cluster_label], alpha=0.5)
    plt.title("d=3")
    plt.show()
    # cut the source matrix into smaller parts
    # chinese displaying
    '''
    label_list = []
    label_list.append(label[0:50])
    label_list.append(label[50:100])
    label_list.append(label[100:150])
    label_list.append(label[150:200])
    label_list.append(label[200:220])
    i = 0
    #group_num = src.shape[0] / 50 + 1
    group_num = 5
    j = 0
    for i in range(group_num):
        sub_label_list = label_list[i]
        for cluster_index in sub_label_list:
            plt.scatter(src[j][0], src[j][1], c=table[cluster_index])
            plt.text(src[j][0], src[j][1], s=names[j], fontproperties=prop, size=6)
            j += 1
        plt.show()
    '''

def silhouette(X, K):
    keys = []
    for i in range(K):
        keys.append(i)
    table = dict(zip(keys, colors.BASE_COLORS))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, np.size(X) + (K + 1) * 10])
    clusterer = KMeans(n_clusters=K, random_state=10)
    clusterer_labels = clusterer.fit_predict(X)
    print(clusterer_labels)
    silhouette_avg = silhouette_score(X, clusterer_labels)
    print("for ", K, "the average silhouette score is: ", silhouette_avg)
    sample_silhouette = silhouette_samples(X, clusterer_labels)
    y_low = 10
    for i in range(K):
        silhouette_val = sample_silhouette[clusterer_labels == i]
        silhouette_val.sort()
        y_high = y_low + silhouette_val.shape[0]
        #color = plt.get_cmap('Spectral')(float(i) / K)
        color = table[i]
        ax1.fill_betweenx(np.arange(y_low, y_high), 0, silhouette_val, facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_low + 0.5 * silhouette_val.shape[0], str(i))
        y_low = y_high + 10
    ax1.axvline(x=silhouette_avg, color='red', linestyle="--")
    plt.show()
    return clusterer_labels


def to_text(name_list, cluster_labels, K):
    f_list = []
    for i in range(K):
        f = open('%d.txt' % i, 'a')
        f_list.append(f)
    for index, label in enumerate(cluster_labels):
        name = name_list[index]
        f = f_list[label]
        f.write(name)
        f.write('\n')


if __name__=="__main__":
    K = int(input("Number of clusters: "))
    data, names = load_data(K)
    print(data)
    # lle visualizing
    #l = LLE(copy.deepcopy(data))
    #show_data_2d, show_data_3d = l.analyze()
    ''' k-means by QHX
    print("data shape:",data.shape)
    print("Before preprocess:",data[0:5, :])
    data=data_pre_processing(data)
    print("After preprocess:",data[0:5,:])
    record_index=run_k_means(data,K)
    print(record_index)
    '''
    pca_2d = PCA(n_components=2).fit_transform(data)
    pca_3d = PCA(n_components=3).fit_transform(data)
    cluster_labels = silhouette(data, K)
    to_text(names, cluster_labels, K)
    visualize(pca_2d, pca_3d, cluster_labels, names, K)
    #visualize(show_data_2d, show_data_3d, cluster_labels, names, K)
    '''
    df=pd.DataFrame({"V1":data[:,0],"V2":data[:,1],"class":record_index})
    fig=plt.figure()
    g=sns.FacetGrid(df,hue="class")
    g.map(plt.scatter,"V1","V2")
    g.add_legend()
    plt.show()
    '''