FROM centos
RUN yum install python3 -y
RUN pip3 --no-cache-dir install numpy
RUN pip3 install --upgrade pip
RUN pip install --upgrade pip
RUN pip install wheel
RUN pip install setuptools
RUN pip install keras
RUN pip install opencv-python
RUN yum install -y libSM
RUN yum install -y libXext
RUN yum install -y libXrender
RUN pip3 install tensorflow
RUN pip3 install --upgrade tensorflow-probability
RUN yum install git -y
RUN pip3 install matplotlib
RUN echo "jenkins ALL=(ALL) NOPASSWD:ALL">>/etc/sudoers


