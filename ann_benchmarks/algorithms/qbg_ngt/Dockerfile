FROM ann-benchmarks

RUN apt update
RUN apt install -y git cmake g++ python3 python3-setuptools python3-pip libblas-dev liblapack-dev
RUN pip3 install wheel pybind11==2.5.0
ADD NGT/ /home/app/ngt/
#RUN git clone -b main https://github.com/yahoojapan/ngt.git
RUN mkdir -p ngt/build
RUN cd ngt/build && cmake -DNGT_AVX2=ON ..
RUN cd ngt/build && make -j 4 && make install
RUN ldconfig
RUN cd ngt/python && python3 setup.py bdist_wheel
RUN pip3 install ngt/python/dist/ngt-*-linux_x86_64.whl
