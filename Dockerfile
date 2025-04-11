FROM balenalib/amd64-debian:bullseye
WORKDIR /root/

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential libssl-dev zlib1g-dev libncurses5-dev libreadline-dev \
    libsqlite3-dev libgdbm-dev libbz2-dev libexpat1-dev liblzma-dev \
    libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran \
    cmake git unzip libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev ccache \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    gstreamer1.0-rtsp gstreamer1.0-vaapi gstreamer1.0-omx v4l2loopback-utils \
    ocl-icd-libopencl1 clinfo aircrack-ng && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/ccache /usr/local/bin/cc

RUN wget https://www.python.org/ftp/python/3.9.14/Python-3.9.14.tgz && \
    tar xzf Python-3.9.14.tgz && \
    cd Python-3.9.14 && \
    ./configure --enable-optimizations --with-lto && \
    make -j$(nproc) altinstall && \
    cd .. && \
    rm -rf Python-3.9.14 Python-3.9.14.tgz

RUN wget https://bootstrap.pypa.io/get-pip.py && python3.9 get-pip.py && rm get-pip.py


COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir grpcio==1.66.1 grpcio-tools==1.66.1 protobuf==5.29.3

RUN git clone https://github.com/opencv/opencv.git && \
    cd opencv && \
    git checkout 4.10.0 && \
    git submodule update --recursive --init && \
    cd .. && \
    git clone https://github.com/opencv/opencv_contrib.git && \
    cd opencv_contrib && \
    git checkout 4.10.0 && \
    cd ../opencv && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    #-D EXTRA_C_FLAGS=-mcpu=cortex-a76 -mfpu=neon-vfpv4 -ftree-vectorize -mfloat-abi=hard \
    #-D EXTRA_CXX_FLAGS=-mcpu=cortex-a76 -mfpu=neon-vfpv4 -ftree-vectorize -mfloat-abi=hard \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_DOCS=OFF \
    -D BUILD_opencv_world=OFF \
    -D BUILD_opencv_python3=ON \
    -D BUILD_opencv_python2=OFF \
    -D PYTHON_EXECUTABLE=$(which python3.9) \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D PYTHON3_EXECUTABLE=$(which python3.9) \
    -D PYTHON3_INCLUDE_DIR=$(python3.9 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.9/site-packages \
    -D WITH_GSTREAMER=ON \
    -D BUILD_EXAMPLES=OFF .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd ../../ && \
    rm -rf opencv opencv_contrib
RUN pip --no-cache-dir install onnxruntime
COPY . .

ENTRYPOINT ["python3.9", "test_pb_main.py"]