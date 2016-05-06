FROM nvidia/cuda:7.5-centos7

MAINTAINER Andrea Peruffo <andrea.peruffo1982@gmail.com>

# --- --- ---  Proxy Settings --- --- ---
#UNCOMMENT IF BHEIND A PROXY
#SET A PROPER PROXY IP

#ENV DOCKER_PROXY YOUR_PROXY_IP

#ENV http_proxy ${DOCKER_PROXY}
#ENV HTTP_PROXY ${DOCKER_PROXY}
#ENV https_proxy ${DOCKER_PROXY}
#ENV HTTPS_PROXY ${DOCKER_PROXY}
#ENV NO_PROXY '127.0.0.1, localhost, /var/run/docker.sock'

# --- --- ---  Update OS --- --- ---
RUN yum -y update; yum clean all

# --- --- ---  Tar --- --- ---
RUN yum -y install tar bzip2

# --- --- ---  Gcc / C++ --- --- ---
RUN yum -y install gcc gcc-c++

# --- --- ---  Make --- --- ---
RUN yum -y install make

# --- --- ---  Gcc compilation deps --- --- ---
RUN yum -y install gmp gmp-devel mpfr mpfr-devel libmpc libmpc-devel

# --- --- ---  Libtool --- --- ---
RUN yum -y install libtool

WORKDIR /home/

# --- --- ---  GCC 4.9 --- --- ---
RUN curl -L "http://gcc.skazkaforyou.com/releases/gcc-4.9.3/gcc-4.9.3.tar.bz2" -o /home/gcc.tar.bz2

RUN tar xvfj /home/gcc.tar.bz2

WORKDIR /home/gcc-4.9.3

RUN ./configure --disable-multilib --enable-languages=c,c++
RUN make
RUN make install

# --- --- ---  NTL --- --- ---
WORKDIR /home

RUN curl -L "http://www.shoup.net/ntl/ntl-9.8.1.tar.gz" -o /home/ntl.tar.gz

RUN tar xvf /home/ntl.tar.gz

WORKDIR /home/ntl-9.8.1/src

RUN ./configure SHARED=on NTL_EXCEPTIONS=on
RUN make
RUN make install

# --- --- ---  Cmake --- --- ---
RUN yum install -y cmake

# Fix cuda GCC
RUN ln -fs /usr/local/bin/gcc /usr/bin/gcc
RUN ln -fs /usr/local/bin/g++ /usr/bin/g++
RUN ln -fs /usr/local/bin/cpp /usr/bin/cpp
RUN ln -fs /usr/local/bin/c++ /usr/bin/c++

#Fix library path
RUN echo "export LD_LIBRARY_PATH=/usr/local/lib64/:/usr/local/lib:${LD_LIBRARY_PATH}" >> ~/.bashrc

WORKDIR /home/sources

CMD ["/bin/bash"]