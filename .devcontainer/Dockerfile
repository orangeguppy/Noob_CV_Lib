FROM mcr.microsoft.com/devcontainers/cpp:1-debian-12

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="none"

RUN sudo apt-get update
RUN sudo apt-get install -y cmake protobuf-compiler libprotobuf-dev libopencv-dev ninja-build