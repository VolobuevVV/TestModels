FROM balenalib/amd64-debian:bullseye

WORKDIR /root/

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential libssl-dev zlib1g-dev libncurses5-dev libreadline-dev \
    libsqlite3-dev libgdbm-dev libbz2-dev libexpat1-dev liblzma-dev \
    libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran \
    cmake git unzip ccache \
    v4l2loopback-utils \
    ocl-icd-libopencl1 clinfo && \
    rm -rf /var/lib/apt/lists/*

RUN echo 'CFLAGS="-march=znver3 -O3 -ftree-vectorize -flto"' >> /etc/environment && \
    echo 'CXXFLAGS="-march=znver3 -O3 -ftree-vectorize -flto"' >> /etc/environment

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
    -D CMAKE_C_FLAGS="-march=znver3 -O3 -ftree-vectorize -flto" \
    -D CMAKE_CXX_FLAGS="-march=znver3 -O3 -ftree-vectorize -flto" \
    # Если ошибка:
    #-D CMAKE_C_FLAGS="-march=znver3 -O3 -ftree-vectorize" \
    #-D CMAKE_CXX_FLAGS="-march=znver3 -O3 -ftree-vectorize" \
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
    -D BUILD_opencv_java=OFF \
    -D BUILD_opencv_js=OFF \
    -D BUILD_opencv_python_bindings_generator=OFF \
    -D BUILD_opencv_python_tests=OFF \
    -D BUILD_opencv_ts=OFF \
    -D BUILD_JAVA=OFF \
    -D BUILD_CUDA_STUBS=OFF \
    -D WITH_QT=OFF \
    -D WITH_GTK=OFF \
    -D WITH_OPENGL=OFF \
    -D WITH_VTK=OFF \
    -D WITH_IPP=OFF \
    -D WITH_TBB=OFF \
    -D WITH_1394=OFF \
    -D WITH_OPENCL=OFF \
    -D WITH_OPENCL_SVM=OFF \
    -D WITH_OPENCLAMDFFT=OFF \
    -D WITH_OPENCLAMDBLAS=OFF \
    -D WITH_WEBP=OFF \
    -D WITH_TIFF=OFF \
    -D WITH_EIGEN=OFF \
    -D WITH_GDCM=OFF \
    -D WITH_JASPER=OFF \
    -D WITH_OPENEXR=OFF \
    -D BUILD_EXAMPLES=OFF .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd ../../ && \
    rm -rf opencv opencv_contrib

RUN pip install prettytable
COPY . .

ENTRYPOINT ["python3.9", "test.py"]
