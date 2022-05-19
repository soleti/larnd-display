# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

ENV version=0.3.1
RUN apt-get update; apt-get install curl -y
RUN curl "https://codeload.github.com/DUNE/larnd-sim/tar.gz/refs/tags/v$version" | tar zxf -
RUN sed -i "s/cupy//g" larnd-sim-${version}/setup.py
RUN cd larnd-sim-${version} && pip install .
COPY . .

RUN pip install .
RUN pip install gunicorn
RUN chmod -R 777 .

# Dockerfile
ARG USER=roberto
ARG UID=92317
ARG GID=92317
# default password for user
ARG PW=docker
RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | \
      chpasswd

# Setup default user, when enter docker container
USER ${UID}:${GID}

# CMD ["gunicorn", "'index:run_display(", '"larnd-sim-0.3.1"', "https://portal.nersc.gov/project/dune/data/)'", "--bind=0.0.0.0:5000"]
CMD gunicorn 'index:run_display("larnd-sim-0.3.1", "https://portal.nersc.gov/project/dune/data/")' --workers=2 --bind=0.0.0.0:5000
# CMD exec ./evd.py larnd-sim-${version} --host 0.0.0.0 --filepath=https://portal.nersc.gov/project/dune/data/
