#from __future__ import absolute_import
#import numpy as np
#import os
#import time
#import subprocess
#from benchmark.algorithms.base import BaseANN
#from benchmark.datasets import DATASETS, download_accelerated
import os
import subprocess
import time

import ngtpy

from ..base.module import BaseANN

class QBG(BaseANN):
    def __init__(self, metric, object_type, epsilon, params):
        self._params = params
        self._metric = metric
        self._first_samples = params.get("Fs", 2000)
        self._second_clusters = params.get("Sc", 1024)
        self._second_samples = params.get("Ss", 100)
        self._third_clusters = params.get("Tc", 10000)
        self._third_samples = params.get("Ts", 10000)
        self._num_of_r_samples = params.get("rs", 1000)
        self._expected_recall = params.get("er", 0.97)
        self._num_of_r_iterations = params.get("ri", 100)
        self._r_step = params.get("rx", 2)
        self._blob_epsilon = params.get("be", 0.05)
        self._clustering = "3"
        #self._clustering = "22"
        self._ngt_root = "data/ngt"  # debug
        #self._ngt_root = "ngt"
        self._ngt_index_root = self._ngt_root + "/indexes/"
        self._is_open = False

    #def setIndexPath(self, dataset):
    #    self._path = f"data/indices/trackT1/algo.NGT:q{self._quantization}-s{self._quantization_sample}-b{self._blob}-rs{self._num_of_r_samples}-ri{self._num_of_r_iterations}"
    #    os.makedirs(self._path, exist_ok=True)
    #    self._index_path = os.path.join(self._path, DATASETS[dataset]().short_name())
        
    def fit(self, X):
        print("QBG: start indexing...")
        dim = len(X[0])
        print("QG: # of data=" + str(len(X)))
        print("QG: dimensionality=" + str(dim))
        #index_dir = "indexes"
        index_dir = "results/indexes"
        if dim <= 128:
            pseudo_dimension = 192
            subvector_dimension = 2
        elif dim <= 256:
            pseudo_dimension = 256
            subvector_dimension = 4
        #self.setIndexPath(dataset)
        #print("QBG: dataset:", dataset)
        #print("QBG: dataset str:", ds.__str__())
        #print("QBG: distance:", ds.distance())
        print("QBG: dimension:", dim)
        #print("QBG: type:", ds.dtype)
        #print("QBG: nb:", ds.nb)
        #print("QBG: dataset file name:", ds.get_dataset_fn())
        print("QBG: # of second clusters (Sc):", self._second_clusters)
        print("QBG: # of second samples (Ss):", self._second_samples)
        print("QBG: # of third clusters (Tc):", self._third_clusters)
        print("QBG: # of third samples (Ts):", self._third_samples)
        print("QBG: # of r samples (rs):", self._num_of_r_samples)
        print("QBG: # of r iterations (ri):", self._num_of_r_iterations)
        print("QBG: # of steps (rx):", self._r_step)
        print("QBG: blob epsilon (be):", self._r_step)
        print("QBG: expected recall (er):", self._expected_recall)
        print("QBG: ngt root:", self._ngt_root)
        print("QBG: index path:", index_dir)

        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        self._index_path = os.path.join(index_dir, "QBG-{}-{}-{}-{}-{}-{}-{}-{}".format(self._first_samples, self._second_clusters, self._second_samples, self._third_clusters, self._third_samples, self._num_of_r_samples, self._num_of_r_iterations, self._r_step))

        if os.path.exists(self._index_path):
            print('QBG: The specified index already exists')
        else:
            print('QBG: create...')
            num_of_subvectors = pseudo_dimension // subvector_dimension
            args = ['qbg', 'create',
                    '-d' + str(dim),
                    '-P' + str(pseudo_dimension),
                    '-of', '-Of', '-D2',
                    '-N' + str(num_of_subvectors),
                    self._index_path]
            print(args)
            subprocess.call(args)
            f = open(self._index_path + "/log", 'w')
            f.write(" ".join(args) + "\n")

            idx = ngtpy.QuantizedBlobIndex(self._index_path, read_only=False)
            idx.batch_insert(X, debug=False)
            idx.save()

            print('QBG: build...')
            if self._clustering == "3":
                c3 = self._third_clusters
                s3 = c3 * self._third_samples
                c2 = self._second_clusters
                s2 = c2 * self._second_samples
                #c1 = math.floor(math.sqrt(c2))
                c1 = c2 / 100
                if c1 > 7000:
                    c1 = 7000
                s1 = c1 * self._first_samples
            else:
                c3 = self._third_clusters
                s3 = c3 * self._third_samples
                c2 = self._second_clusters
                s2 = c2 * self._second_samples
                #c1 = math.floor(math.sqrt(c2))
                c1 = c3 / 100
                if c1 > 7000:
                    c1 = 7000
                s1 = c1 * self._first_samples
            args = ['qbg', 'build',
                    '-p1-2',
                    '-B' + self._clustering + ',' + 
                    str(s1) + ':' + str(c1) + ',' +
                    str(s2) + ':' + str(c2) + ',' +
                    str(s3) + ':' + str(c3),
                    '-O' + str(len(X)),
                    '-o' + str(self._num_of_r_samples),
                    '-X' + str(self._r_step),
                    '-I' + str(self._num_of_r_iterations),
                    '-A' + str(self._expected_recall),
                    '-Pr' , '-M1', '-ip', '-S2', '-t600', '-v',
                    self._index_path]
            print(args)
            subprocess.call(args)
            f.write(" ".join(args) + "\n")
            args = ['qbg', 'build',
                    '-p3', '-v',
                    self._index_path]
            print(args)
            subprocess.call(args)
            f.write(" ".join(args) + "\n")
            f.close
        
        if not self._is_open:
            print("QBG: opening the index...")
            self._index = ngtpy.QuantizedBlobIndex(self._index_path)
            self._is_open = True

    def set_query_arguments(self, query_args):
        self._epsilon = query_args.get("epsilon", 0.1)
        self._edge_size = query_args.get("edge", 0)
        self._exploration_size = query_args.get("blob", 120)
        self._num_of_probes = query_args.get("probe", 0)
        self._blob_epsilon = query_args.get("blob_epsilon", self._blob_epsilon)
        # only this part is different between t1 and t2
        #self._exact_result_expansion = query_args.get("expansion", 2.0)
        self._exact_result_expansion = 0.0
        #self._function_selector = 4 + 3 #######################
        #self._function_selector = 0 ############################
        self.name = "QBG-NGT(%1.3f, %s, %s, %s, %1.3f, %1.3f)" % (
            self._epsilon,
            self._edge_size,
            self._exploration_size,
            self._num_of_probes,
            self._exact_result_expansion,
            self._blob_epsilon,
        )
        self._index.set_with_distance(False)
        self._index.set(epsilon=self._epsilon,
                        blob_epsilon=self._blob_epsilon,
                        edge_size=self._edge_size,
                        exploration_size=self._exploration_size,
                        exact_result_expansion=self._exact_result_expansion,
                        #function_selector=self._function_selector, ####################
                        num_of_probes=self._num_of_probes)
        print(f"QBG: eps={self._epsilon} blobeps={self._blob_epsilon} edge={self._edge_size} blob={self._exploration_size}")

    def query(self, X, n):
        return self._index.search(X, n)

    #def __str__(self):
    #    return "NGT-T1:" + self.getTitle() + f"-be{self._blob_epsilon}-e{self._epsilon}-bl{self._exploration_size}-p{self._num_of_probes}"
