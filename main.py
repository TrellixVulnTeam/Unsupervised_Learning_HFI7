#%%

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import SparsePCA

from scipy import stats
from scipy.stats import multivariate_normal
import math
import os
import random

# Included following due to internet certificate problems
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#%%

## Move to correct folder for server.  Can remove before sending
# os.chdir('/home/poblivsig/Dropbox/horses2')
os.chdir('/Users/paullivesey/Dropbox/2. Personal/3. Projects/Python/unsupervised')

print(os.getcwd())

#%%

## Open the pre-processed csv
df = pd.read_csv('data/winequality-red.csv')
# df = pd.read_csv('data/phishing.csv')

#%%

## Get info about wine
print(f'Shape\n\n{df.shape}')
print(f'Columns\n\n{df.columns}')
print(f'dtypes\n\n{df.dtypes}')
pd.set_option('display.max_columns', None)
print(f'Description\n\n{df.describe()}')
print(f'Info:\n{df.info}')
print(f'Check out the sample: {df.sample(n=1)}')
pd.set_option('display.max_columns', 5)


#%%

y = df['quality']
X = df.drop('quality', axis=1)
print(df.shape)

#%% Constants for different algorithms

N_CLUSTERS = 10
GM_N_CLUSTERS = 10
INIT = 'k-means++'
N_INIT = 10
KM_MAX_ITERS = 300
TOLERANCE = 1e-4
PC_DISTANCES = True
KM_VERBOSE = 0
KM_RANDOM_STATE = 42
ALGORITHM = 'full'
FEATURE_1_TO_PLOT = 8
FEATURE_2_TO_PLOT = 9
N_COMPONENTS = 11

#%%

# Scale the features (attributes)
scaler = RobustScaler()
X = scaler.fit_transform(X)

#%%

# Visualization of the raw data
sns.set_context('notebook')
plt.style.use('bmh')


#%%

N_CLUSTERS = 10
silhouettes = []

## Loop through the cluster numbers and output silhouette
## and elbow charts

for n_cluster in range(2, N_CLUSTERS+1):
    km = KMeans(n_clusters=n_cluster,
                init=INIT,
                n_init=N_INIT,
                max_iter=KM_MAX_ITERS,
                tol=TOLERANCE,
                precompute_distances=PC_DISTANCES,
                verbose=KM_VERBOSE,
                random_state=KM_RANDOM_STATE,
                algorithm=ALGORITHM)

    print(f'inertia for {n_cluster} clusters = {km.inertia_}')
    y_pred = km.fit_predict(X)

    ### Print some stats
    print(f'inertia = {km.inertia_}')
    silhouettes.append(silhouette_score(X, km.labels_, metric='euclidean'))
    # print(f'silhouette score = {s_score:.3f}')

print(f'silhouettes = {silhouettes}')


#%%

sns.lineplot(x=np.arange(2 ,N_CLUSTERS+1), y=silhouettes)

#%%
def kmeans(Xk, xlim, ylim, data_title):
    print(len(np.arange(2, N_CLUSTERS+1)))
    print(len(silhouettes))

    #%
    #****** Run the KMeans and create Silhouette and scatter ******
    clusters = np.arange(2, N_CLUSTERS+1)
    silhouette_scores = {}

    ## Borrowed from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    for cluster in clusters:
        ## Build the plots
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ## The plot on x for the silhouette coeffients ranges from -1 to +1
        ax1.set_xlim([-0.25, 1])
        ## The plot on y has to include all of the shapes with their values sorted
        ax1.set_ylim([0, len(X) + (cluster + 1) * 10])
        fig.set_size_inches(16, 6)

        ## Now run the clustering algorithm itself
        km = KMeans(n_clusters=cluster,
                    init=INIT,
                    n_init=N_INIT,
                    max_iter=KM_MAX_ITERS,
                    tol=TOLERANCE,
                    precompute_distances=PC_DISTANCES,
                    verbose=KM_VERBOSE,
                    random_state=KM_RANDOM_STATE,
                    algorithm=ALGORITHM)

        y_pred = km.fit_predict(Xk)
        cluster_lbls = km.labels_

        ## Get the silhoueete score which gives a basic silhouette_score
        ## for the run.  Store away from plotting later
        silhouette_average = silhouette_score(Xk, y_pred)
        silhouette_scores[cluster] = silhouette_average
        # What is the silhouette score for each instance?
        sample_silhouette_scores = silhouette_samples(Xk, y_pred)

        lower_y = 10
        for j in range(cluster):
            # Group together the silhouette coefficients for cluster i
            # and the sort them from largest to smallest
            j_cluster_coeffs = sample_silhouette_scores[y_pred == j]
            j_cluster_coeffs.sort()

            ## Get bottom of cluster shape for chart
            upper_y = lower_y + j_cluster_coeffs.shape[0]
            colour = cm.rainbow(float(j) / cluster)

            ## Draw the cluster shape
            ax1.fill_betweenx(np.arange(lower_y, upper_y),
                             0, j_cluster_coeffs,
                             facecolor=colour, edgecolor=colour, alpha=0.7)
            ax1.text(-0.05, lower_y + 0.5 *j_cluster_coeffs.shape[0], str(j))

            # Get the next clusters position
            lower_y = upper_y + 10

        ## Draw the average silhouette score line.
        ax1.axvline(x=silhouette_average, color="green", linestyle="--")

        ## Set the title and labels
        ax1.set_xlabel('Silhouette Coefficient', fontsize=11)
        ax1.set_ylabel('Cluster', fontsize=11)

        ax1.set_title(f'Silhouette Diagram for {cluster} Clusters', fontsize=14)

        ## Create 2D scatterplot for the clusters created above
        ax2.scatter( Xk[:, FEATURE_1_TO_PLOT],
                     Xk[:, FEATURE_2_TO_PLOT],
                     marker='.',
                     s=30,
                     lw=0,
                     alpha=0.5,
                     c=cm.rainbow(km.labels_.astype(float) / cluster),
                     edgecolor='k')
        ax2.scatter(km.cluster_centers_[:, 0],
                    km.cluster_centers_[:, 1],
                    marker='o',
                    c='white',
                    alpha=1,
                    s=180,
                    edgecolor='k')

        for i, c in enumerate(km.cluster_centers_):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
        ax2.set_xlim([0, xlim])
        ax2.set_xlim([0, ylim])
        ax2.set_xlabel('1st Feature')
        ax2.set_ylabel('2nd Feature')
        ax2.set_title(f'{cluster} Cluster data scatterplot for 2 Features.', fontsize=14)

        plt.suptitle((f'{data_title} K-Means Clustering on Sample Data with {cluster} Clusters'),
                     fontsize=15)

    # print(f'silhouette scores = {silhouette_scores}')
    plt.show()
    print(f'silhouettes = {silhouettes}')

    plt.title(f'Silhouette Line-plot for {data_title}', fontsize=14)
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette Score')
    plt.plot(np.arange(1, 10),silhouettes)
    plt.show()


#%%
#g****** Run the GAUSSIAN MIXTURE and create Silhouette and scatter ******
clusters = np.arange(2, N_CLUSTERS+1)
silhouette_scores = {}
bics = []
aics = []

## Borrowed from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
for cluster in clusters:
    ## Build the plots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ## The plot on x for the silhouette coeffients ranges from -1 to +1
    ax1.set_xlim([-0.25, 1])
    ## The plot on y has to include all of the shapes with their values sorted
    ax1.set_ylim([0, len(X) + (cluster + 1) * 10])
    fig.set_size_inches(16, 6)

    ## Now run the clustering algorithm itself
    gm = GaussianMixture(n_components=cluster )
    y_pred_gm = gm.fit_predict(X)
    bics.append(gm.bic(X))
    aics.append(gm.aic(X))
    # km = KMeans(n_clusters=cluster,
    #             init=INIT,
    #             n_init=N_INIT,
    #             max_iter=KM_MAX_ITERS,
    #             tol=TOLERANCE,
    #             precompute_distances=PC_DISTANCES,
    #             verbose=KM_VERBOSE,
    #             random_state=KM_RANDOM_STATE,
    #             algorithm=ALGORITHM)

    # y_pred = km.fit_predict(X)
    cluster_lbls = np.unique(y_pred_gm[:cluster]) #gm.labels_

    ## Get the silhouette score which gives a basic silhouette_score
    ## for the run.  Store away from plotting later
    silhouette_average = silhouette_score(X, y_pred_gm)
    silhouette_scores[cluster] = silhouette_average
    # What is the silhouette score for each instance?
    sample_silhouette_scores = silhouette_samples(X, y_pred_gm)

    lower_y = 10
    for j in range(cluster):
        # Group together the silhouette coefficients for cluster i
        # and the sort them from largest to smallest
        j_cluster_coeffs = sample_silhouette_scores[y_pred_gm == j]
        j_cluster_coeffs.sort()

        ## Get bottom of cluster shape for chart
        upper_y = lower_y + j_cluster_coeffs.shape[0]
        colour = cm.rainbow(float(j) / cluster)

        ## Draw the cluster shape
        ax1.fill_betweenx(np.arange(lower_y, upper_y),
                         0, j_cluster_coeffs,
                         facecolor=colour, edgecolor=colour, alpha=0.7)
        ax1.text(-0.05, lower_y + 0.5 *j_cluster_coeffs.shape[0], str(j))

        # Get the next clusters position
        lower_y = upper_y + 10

    ## Draw the average silhouette score line.
    ax1.axvline(x=silhouette_average, color="green", linestyle="--")

    ## Set the title and labels
    ax1.set_xlabel('Silhouette Coefficient', fontsize=11)
    ax1.set_ylabel('Cluster', fontsize=11)

    ax1.set_title(f'Silhouette Diagram for {cluster} Clusters', fontsize=14)

    ## Create 2D scatterplot for the clusters created above
    ax2.scatter( X[:, FEATURE_1_TO_PLOT],
                 X[:, FEATURE_2_TO_PLOT],
                 marker='.',
                 s=30,
                 lw=0,
                 alpha=0.5,
                 c=cm.rainbow(km.labels_.astype(float) / cluster),
                 edgecolor='k')
    ## Find centers for Gaussian clusters (choosing the points with
    ## the maximal density to represent its cluster.
    centers = np.empty(shape=(gm.n_components, X.shape[1]))
    for i in range(gm.n_components):
        density = multivariate_normal(cov=gm.covariances_[i],
                                      mean=gm.means_[i]).logpdf(X)
        centers[i, :] = X[np.argmax(density)]

    ax2.scatter(centers[:, 0],
                centers[:, 1],
                marker='o',
                c="white",
                alpha=1,
                s=180,
                edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0],
                    c[1],
                    marker='$%d$' % i,
                    alpha=1,
                    s=50,
                    edgecolor='k')
    ax2.set_xlim([0,5])
    ax2.set_xlabel("1st Feature")
    ax2.set_ylabel("2nd Feature")
    ax2.set_title("Clustered data scatterplot for 2 Features.", fontsize=14)

    plt.suptitle((f'K-Means Clustering on Sample Data with {cluster} Clusters'),
                 fontsize=14)

# print(f'silhouette scores = {silhouette_scores}')
plt.show()





#%%
## Create the Silhouette Score Chart
sns.lineplot(x=clusters, y=list(silhouette_scores.values()))

#%%
bics = []
aics = []
clusters = np.arange(2, N_CLUSTERS+1)

for n_cluster in clusters:
    gm = GaussianMixture(n_components=n_cluster )
    y_clust_gm = gm.fit_predict(X)
    bics.append(gm.bic(X))
    aics.append(gm.aic(X))

#%%
## Create AIC and BIC chart
plt.plot(clusters, bics, label='bic')
plt.plot(clusters, aics, label='aic')
plt.legend()
plt.show()
#%%
## Find the Bayesian Gaussian Mixture
bgm = BayesianGaussianMixture ( n_components = 10 , n_init = 10 )
bgm.fit ( X )
print(np.round ( bgm.weights_ , 2 ))

##
#%%
## Calculate the best PCS dimensions
# pca_res = KernelPCA(n_components=6, kernel='rbf', degree=4, gamma=0.1)
# pca_res.fit(X)
# d = np.argmax(np.cumsum(pca_res.explained_variance_ratio_) >=  0.95) + 1
# print(f'optimal PCA dimensions = {d}')

#%%
## Use the optimal dimension to calculate the principal components...

# pca = KernelPCA(n_components=5)
# X2dim = pca.fit_transform(X)
# print(f'New dimensions = {X2dim.shape}')
# print(f'principal components = {pca.explained_variance_ratio_}')

#%%
## Plot different dimensions against the explained variance
NO_DIMS_TO_CHECK = 11
dimensions = np.arange(1, NO_DIMS_TO_CHECK)
expl_variances = []

for dimension in dimensions:
    pca = PCA(n_components=dimension)
    pca.fit(X)
    expl_variances.append(np.sum(pca.explained_variance_ratio_))

print(f'expl_variances = \n{expl_variances}')
print(f'dimensions = \n{dimensions}')
plt.title('Explained Variance vs. Dimensions', fontsize=14)
plt.xlabel('No. of Dimensions', fontsize=12)
plt.ylabel('Explained Variance', fontsize=12)
plt.axvline(6, color='r', linestyle='dotted')
plt.axvline(6, color='r', linestyle='dotted')
plt.axhline(0.938, color='r', linestyle='dotted')
sns.lineplot(dimensions, expl_variances)
#%%
def CA_Algorithm_2D(algorithm, KPCA, PCA_TYPE, title, components, **args):
    if KPCA:
        result = algorithm(n_components=components,
                           kernel=args['kernel'],
                           degree=args['degree'],
                           gamma=args['gamma'])
    else:
        result = algorithm(n_components=components) #, kernel='rbf', degree=4, gamma=0.4)

    X2dim = result.fit_transform(X)
    print(f'New dimensions = {X2dim.shape}')

    QUALITY_1 = 5
    QUALITY_2 = 6
    QUALITY_3 = 7

    qualities = [QUALITY_1,
                 QUALITY_2,
                 QUALITY_3]
    plt.figure()
    for col, j in zip (['r', 'c', 'y'], qualities):
        plt.scatter(X2dim[y == j, 0],
                    X2dim[y == j, 1],
                    alpha=0.4,
                    marker='.',
                    label=j,
                    color=col, #['r', 'c'],
                    lw=2,
                    s=40)
    plt.legend(title='Quality', loc='best')
    plt.xlabel('X1', fontsize=12)
    plt.ylabel('X2', fontsize=12)
    plt.title(title)
    plt.show()

    if PCA_TYPE:
        d = np.argmax(np.cumsum(result.explained_variance_ratio_) >=  0.95) + 1
        print(f'optimal PCA dimensions = {d}')

    ## Return the results to be used in other algorithms
    return X2dim

#%%
def CA_Algorithm_3D(algorithm, KPCA, title, components, **args):
    if KPCA:
        result = algorithm(n_components=components,
                           kernel=args['kernel'],
                           degree=args['degree'],
                           gamma=args['gamma'])
    else:
        result = algorithm(n_components=components) #, kernel='rbf', degree=4, gamma=0.4)

    ## ICA - Using 3 dimensions for visual analysis
    X3dim = result.fit_transform(X)
    print(f'New dimensions = {X3dim.shape}')

    QUALITY_1 = 5
    QUALITY_2 = 6
    QUALITY_3 = 7
    qualities = [QUALITY_1,
                 QUALITY_2,
                 QUALITY_3]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection='3d')
    for col, j in zip (['r', 'c', 'y'], qualities):
        ax.scatter(X3dim[y == j, 0],
                   X3dim[y == j, 1],
                   X3dim[y == j, 2],
                   alpha=0.4,
                   marker='.',
                   label=j,
                   color=col,
                   lw=2,
                   s=60)
    plt.legend(title='Quality', loc='best')
    # plt.xlabel('X1', fontsize=12)
    # plt.ylabel('X2', fontsize=12)
    # plt.ylabel('other', fontsize=12)
    plt.title(title, fontsize=14)
    plt.show()

    ## Return the results to be used in other algorithms
    return X3dim

#%%
# Run KMeans with X
kmeans(X, 3, 3, 'Original Data')

#%%
# Build PCA Charts
Xpca = CA_Algorithm_2D(PCA,
                       False,
                       True,
                       'Scatterplot for PCA reduction to 2D',
                       components=N_COMPONENTS)

CA_Algorithm_3D(PCA,
                False,
                'Scatterplot for PCA reduction to 3D',
                components=N_COMPONENTS)

#%%
## Run KMeans again with reduced dimension data
kmeans(Xpca, 2, 2, 'PCA Data')

#%%
# Build Kernel PCA Charts
Xkpca = CA_Algorithm_2D(KernelPCA,
                        True,
                        True,
                        'Scatterplot for Kernel PCA reduction to 2D',
                        components=N_COMPONENTS,
                        kernel='rbf',
                        degree=4,
                        gamma=0.4)
CA_Algorithm_3D(KernelPCA,
                True,
                'Scatterplot for Kernel PCA reduction to 3D',
                components=N_COMPONENTS,
                kernel='rbf',
                degree=4,
                gamma=0.4)
#%%
## Run KMeans again with reduced dimension data
kmeans(Xkpca, 0.5, 0.5, 'Kernel PCA Data')

#%%
# Build ICA Charts
Xica = CA_Algorithm_2D(FastICA,
                       False,
                       False,
                       'Scatterplot for ICA reduction to 2D',
                       components=N_COMPONENTS)

CA_Algorithm_3D(FastICA,
                False,
                ' Scatterplot for ICA reduction to 3D',
                components=N_COMPONENTS)

#%%
## Run KMeans again with reduced dimension data
kmeans(Xica, 3, 0.1, 'ICA Data')

#%%
# Build SparsePCA Charts
Xspca = CA_Algorithm_2D(SparsePCA,
                        False,
                        False,
                        'Scatterplot for SparsePCA reduction to 2D',
                        components=N_COMPONENTS)

CA_Algorithm_3D(SparsePCA,
                False,
                'Scatterplot for SparsePCA reduction to 3D',
                components=N_COMPONENTS)

#%%
## Run KMeans again with reduced dimension data
kmeans(Xspca, 2.5, 4, 'Sparse ICA Data')
#%%


