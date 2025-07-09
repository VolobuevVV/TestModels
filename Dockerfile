FROM balenalib/amd64-debian:bullseye
WORKDIR /root/

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential libssl-dev zlib1g-dev libncurses5-dev libreadline-dev \
    libsqlite3-dev libgdbm-dev libbz2-dev libexpat1-dev liblzma-dev \
    libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran \
    cmake git unzip v4l2loopback-utils \
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

RUN pip install ultralytics tensorflow

COPY . .

ENTRYPOINT ["python3.9", "test.py"]