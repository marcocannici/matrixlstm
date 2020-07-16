FROM tensorflow/tensorflow:1.12.0-gpu
RUN apt-get update --fix-missing; \
    apt-get install -y libsm6 libxext6 libxrender-dev git
RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install sacred==0.7.5
RUN pip install colorterm
RUN pip install pymongo
RUN pip install h5py
RUN pip install pathlib

RUN mkdir /.config
RUN chmod 777 /.config
RUN mkdir /.local
RUN chmod 777 /.local
RUN mkdir /.cache
RUN chmod 777 /.cache

RUN pip install configargparse pyyaml
RUN apt-get install -y ffmpeg
RUN apt-get install -y python-tk

CMD ["bash"]
WORKDIR /exp
