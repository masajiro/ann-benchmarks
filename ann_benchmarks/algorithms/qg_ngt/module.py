import os
import subprocess
import time
import shutil

import ngtpy

from ..base.module import BaseANN


class QG(BaseANN):
    def __init__(self, metric, object_type, epsilon, param):
        metrics = {"euclidean": "2", "angular": "E"}
        self._edge_size = int(param["edge"])
        self._outdegree = int(param["outdegree"])
        self._indegree = int(param["indegree"])
        self._outdegree_ext = int(param["outdegreeExt"]) if "outdegreeExt" in param.keys() else 0
        self._indegree_ext = int(param["indegreeExt"]) if "indegreeExt" in param.keys() else 0
        self._max_edge_size = int(param["max_edge"]) if "max_edge" in param.keys() else 128
        self._metric = metrics[metric]
        self._object_type = object_type
        self._edge_size_for_search = int(param["search_edge"]) if "search_edge" in param.keys() else -2
        self._tree_disabled = (param["tree"] is False) if "tree" in param.keys() else False
        self._build_time_limit = float(param["timeout"]) if "timeout" in param.keys() else 4
        self._epsilon = float(param["epsilon"]) if "epsilon" in param.keys() else epsilon
        self._sample = int(param["sample"]) if "sample" in param.keys() else 20000
        self._leaf = param["leaf"] if "leaf" in param.keys() else '100:5'
        self._seed = param["seed"] if "seed" in param.keys() else 'f36'
        self._hop = param["hop"] if "hop" in param.keys() else '3:10'
        self._reconst = param["reconst"] if "reconst" in param.keys() else 'base'
        self._refine_k = int(param["refine_k"]) if "refine_k" in param.keys() else 0
        print("QG: edge_size=" + str(self._edge_size))
        print("QG: outdegree=" + str(self._outdegree))
        print("QG: indegree=" + str(self._indegree))
        print("QG: edge_size_for_search=" + str(self._edge_size_for_search))
        print("QG: epsilon=" + str(self._epsilon))
        print("QG: metric=" + metric)
        print("QG: object_type=" + object_type)

    def fit(self, X):
        print("QG: start indexing...")
        dim = len(X[0])
        print("QG: # of data=" + str(len(X)))
        print("QG: dimensionality=" + str(dim))
        index_dir = "indexes"
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        index = os.path.join(index_dir, "ONNG-{}-{}-{}".format(self._edge_size, self._outdegree, self._indegree))
        anngIndex = os.path.join(index_dir, "ANNG-" + str(self._edge_size))
        tempIndex = os.path.join(index_dir, "TEMP-" + str(self._edge_size))
        forestIndex = os.path.join(index_dir, "FOREST-" + str(self._edge_size))
        index = forestIndex
        print("QG: index=" + index)
        if (not os.path.exists(index)) and (not os.path.exists(anngIndex)):
            print("QG: create ANNG")
            t = time.time()
            args = [
                "ngt",
                "create",
                "-p8",
                "-b500",
                "-ga",
                "-oauto",
                "-D" + self._metric,
                "-d" + str(dim),
                "-E" + str(self._edge_size),
                "-e" + str(self._epsilon),
                "-rd",
                "-L" + self._leaf,
                "-s" + self._seed,
                anngIndex,
            ]
            print(" ".join(args))
            subprocess.call(args)
            idx = ngtpy.Index(path=anngIndex)
            idx.batch_insert(X, num_threads=24, build=False, debug=False)
            idx.save()
            idx.close()
            print("QG: build ANNG")
            idx = ngtpy.Index(path=anngIndex)
            idx.build_index();
            idx.save()
            idx.close()
            print("QG: ANNG construction time(sec)=" + str(time.time() - t))
            if self._refine_k >= 0:
                print("QG: create RANNG")
                t = time.time()
                args = [
                    "ngt",
                    "refine-anng",
                    "-e" + str(0.1 if self._refine_k == 0 else self._epsilon),
                    "-k-" + str(self._refine_k),
                    anngIndex,
                    tempIndex,
                ]
                print(" ".join(args))
                subprocess.call(args)
                shutil.rmtree(anngIndex)
                os.rename(tempIndex, anngIndex)
                print("QG: RANNG construction time(sec)=" + str(time.time() - t))
        if not os.path.exists(index):
            print("QG: construct Forest")
            t = time.time()
            args = [
                "ngt",
                "construct-forest",
                "-EH",
                "-H" + self._hop,
                "-ms",
                "-Mg",
                "-o" + str(self._outdegree),
                "-i" + str(self._indegree),
                "-O" + str(self._outdegree_ext),
                "-I" + str(self._indegree_ext),
                "-e0.0",
                forestIndex,
                anngIndex,
            ]
            print(" ".join(args))
            subprocess.call(args)
            print("QG: construct Forest time(sec)=" + str(time.time() - t))
            if self._reconst == "none":
                print("QG: degree adjustment none")
            elif self._reconst == "base":
                print("QG: degree adjustment")
                t = time.time()
                args = [
                    "ngt",
                    "reconstruct-graph",
                    "-mS",
                    "-sp",
                    forestIndex,
                    tempIndex,
                ]
                print(" ".join(args))
                subprocess.call(args)
                shutil.rmtree(forestIndex)
                os.rename(tempIndex, forestIndex)
            else:
                args = [
                    "ngt",
                    "reconstruct-graph",
                    "-mS",
                    "-sp",
                    "-Ps",
                    "-R" + self._reconst,
                    forestIndex,
                    tempIndex,
                ]
                print(" ".join(args))
                subprocess.call(args)
                shutil.rmtree(forestIndex)
                os.rename(tempIndex, forestIndex)
            print("QG: degree adjustment time(sec)=" + str(time.time() - t))
        if not os.path.exists(index + "/qg"):
            print("QG:create and append...")
            t = time.time()
            args = [
                "qbg",
                "create-qg",
                "-R-:u",
                "-k0",
                index]
            print(" ".join(args))
            subprocess.call(args)
            print("QG: create qg time(sec)=" + str(time.time() - t))
            print("QB: build...")
            t = time.time()
            args = [
                "qbg",
                "build-qg",
                "-o" + str(self._sample),
                "-M1",
                "-ib",
                "-I400",
                "-Gz",
                "-Pn",
                "-E" + str(self._max_edge_size),
                index,
            ]
            print(" ".join(args))
            subprocess.call(args)
            print("QG: build qg time(sec)=" + str(time.time() - t))
        if os.path.exists(index + "/qg/grp"):
            print("QG: index already exists! " + str(index))
            t = time.time()
            self.index = ngtpy.QuantizedIndex(index, self._max_edge_size)
            self.index.set_with_distance(False)
            self.indexName = index
            print("QG: open time(sec)=" + str(time.time() - t))
        else:
            print("QG: something wrong.")
        print("QG: end of fit")

    def set_query_arguments(self, parameters):
        result_expansion, epsilon = parameters
        print("QG: result_expansion=" + str(result_expansion))
        print("QG: epsilon=" + str(epsilon))
        self.name = "QG-NGT(%s,%s,%s:%s,%s:%s,%s,%s,%s,%s,%s,%s,%s,%s)" % (
            self._edge_size,
            self._epsilon,
            self._outdegree,
            self._indegree,
            self._outdegree_ext,
            self._indegree_ext,
            self._max_edge_size,
            self._sample,
            self._leaf,
            self._seed,
            self._hop,
            self._refine_k,
            epsilon,
            result_expansion,
        )
        epsilon = epsilon - 1.0
        self.index.set(epsilon=epsilon, result_expansion=result_expansion)

    def query(self, v, n):
        return self.index.search(v, n)

    def freeIndex(self):
        print("QG: free")
