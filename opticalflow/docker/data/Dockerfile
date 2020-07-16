FROM ros:lunar-ros-base
RUN apt-get update; apt-get install -y ros-lunar-cv-bridge
RUN apt-get update; apt-get install -y ros-lunar-vision-opencv
RUN apt-get install -y python-pip
RUN pip install tensorflow==1.10.0
RUN pip install h5py
RUN pip install --upgrade pip
RUN pip install matplotlib
RUN pip install scipy
RUN pip install catkin_pkg rospkg

RUN mkdir /.config
RUN chmod 777 /.config

