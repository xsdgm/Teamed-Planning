FROM osrf/ros:noetic-desktop-full

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential cmake ninja-build pkg-config \
    python3-pip python3-dev python3-numpy python3-scipy python3-skimage \
    python3-networkx python3-opencv python3-jinja2 python3-empy python3-toml \
    python3-yaml python3-packaging python3-pygments python3-six \
    libeigen3-dev libgazebo11-dev libopencv-dev protobuf-compiler \
    libprotobuf-dev libgstreamer-plugins-base1.0-dev libxml2-dev libxml2-utils \
    ros-noetic-mavros ros-noetic-mavros-extras ros-noetic-cv-bridge \
    ros-noetic-tf ros-noetic-tf2-ros ros-noetic-gazebo-ros-pkgs \
    ros-noetic-velodyne-gazebo-plugins \
    xvfb x11vnc novnc websockify fluxbox xterm mesa-utils \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir \
    'empy==3.3.4' 'jinja2==3.0.3' 'numpy==1.24.4' \
    'scipy==1.10.1' 'scikit-image==0.21.0' 'networkx==3.1' \
    'kconfiglib==14.1.0' 'pyros-genmsg==0.5.8' 'packaging==23.2' \
    'toml==0.10.2' 'jsonschema==4.19.2' 'cerberus==1.3.5'
RUN timeout 180 geographiclib-get-geoids egm96-5 \
    && test -s /usr/share/GeographicLib/geoids/egm96-5.pgm
RUN python3 -m pip install --no-cache-dir 'future==0.18.3' 'lxml==4.9.4'

ARG PX4_REF=v1.13.2
ARG BUILD_JOBS=2
RUN git clone --depth 1 --branch "${PX4_REF}" https://github.com/PX4/PX4-Autopilot.git /opt/PX4-Autopilot \
    && cd /opt/PX4-Autopilot \
    && git submodule update --init --recursive --depth 1 \
        Tools/sitl_gazebo src/modules/mavlink/mavlink src/drivers/gps/devices \
        src/lib/events/libevents src/drivers/uavcan/libuavcan
RUN cd /opt/PX4-Autopilot \
    && DONT_RUN=1 make -j"${BUILD_JOBS}" j="${BUILD_JOBS}" px4_sitl_default gazebo

COPY docker/ros1/ /opt/teamed-docker/
RUN sed -i 's/\r$//' /opt/teamed-docker/*.sh \
    && chmod +x /opt/teamed-docker/*.sh \
    && python3 /opt/teamed-docker/add_sensors.py \
    && python3 /opt/PX4-Autopilot/Tools/sitl_gazebo/scripts/jinja_gen.py \
        --stdout --mavlink_id=1 \
        /opt/PX4-Autopilot/Tools/sitl_gazebo/models/iris/iris.sdf.jinja \
        /opt/PX4-Autopilot/Tools/sitl_gazebo > /tmp/iris-check.sdf \
    && source /opt/teamed-docker/environment.sh \
    && timeout 30 gz sdf -k /tmp/iris-check.sdf \
    && rm /tmp/iris-check.sdf
ENV DISPLAY=:1 LIBGL_ALWAYS_SOFTWARE=1 QT_X11_NO_MITSHM=1 \
    ROS_MASTER_URI=http://localhost:11311 ROS_IP=127.0.0.1 \
    PYTHONPATH=/ws/src/teamed-planning/scripts
WORKDIR /ws
ENTRYPOINT ["/opt/teamed-docker/entrypoint.sh"]
CMD ["sim"]
