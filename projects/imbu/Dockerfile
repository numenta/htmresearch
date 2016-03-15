# ----------------------------------------------------------------------
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# See https://hub.docker.com/r/numenta/nupic for details of official public
# build of image from https://github.com/numenta/nupic/blob/master/Dockerfile
FROM numenta/nupic:latest

# OS-level dependencies
RUN apt-get update && apt-get -y install  apt-transport-https \
                                          build-essential \
                                          nginx \
                                          python-matplotlib \
                                          python-software-properties \
                                          software-properties-common

# Install latest Node JS
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv 68576280 && \
    apt-add-repository 'deb https://deb.nodesource.com/node_4.x precise main' && \
    apt-get update && apt-get -y install nodejs

# Pass real Cortical.io API key using '-e CORTICAL_API_KEY=$CORTICAL_API_KEY'
ENV CORTICAL_API_KEY=OVERRIDE_ME
# Select the retina to use  using '-e IMBU_RETINA_ID=$IMBU_RETINA_ID'
ENV IMBU_RETINA_ID=OVERRIDE_ME
# Default cache dir in container
ENV MODEL_CACHE_DIR=/opt/numenta/imbu/cache
# Enable development-mode htmresearch (optional)
ENV PYTHONPATH=/opt/numenta/nupic.research:${PYTHONPATH}

# Listen on container port 80
EXPOSE 80

# Copy htmresearch to /opt/numenta/nupic.research/htmresearch.  Changes to
# files anchored here will trigger cascading creation of dependent file-system
# layers during `docker build`
RUN mkdir -p /opt/numenta/nupic.research
COPY htmresearch/ /opt/numenta/nupic.research/htmresearch
COPY setup.py /opt/numenta/nupic.research/
COPY requirements.txt /opt/numenta/nupic.research/

# Set working directory to /opt/numenta/imbu for subsequent build instructions
WORKDIR /opt/numenta/nupic.research

# Install python package and dependencies
RUN python setup.py develop

# Copy Imbu and nlp-relevant files to their respective locations
COPY projects/imbu/ /opt/numenta/nupic.research/projects/imbu
COPY projects/nlp/ /opt/numenta/nupic.research/projects/nlp
COPY tests/ /opt/numenta/nupic.research/tests

# Set working directory to /opt/numenta/imbu for subsequent build instructions
WORKDIR /opt/numenta/nupic.research/projects/imbu

# Install python package and dependencies
RUN python setup.py develop

# Install javascript packages
RUN npm install

# Build javascript app using npm
RUN npm run build

# Setup persistent model cache. In order to specify a cache-dir outside of the
# container, use `-v <path to cache>:/opt/numenta/imbu/cache` arguments to
# `docker run`, where `<path to cache>` is a location on the docker host.  In
# practice, this is `/imbu/cache`, which resides on an EBS volume attached to
# the production EC2 instance.
RUN mkdir -p cache
VOLUME cache

# Imbu startup command.  This will re-install htmresearch (if necessary), start
# nginx and supervisor.
CMD ./start_imbu.sh

# User may map /opt/numenta/imbu/gui to a local dir for development purposes,
# allowing for modification of gui files without requiring rebuild of image
VOLUME /opt/numenta/nupic.research/projects/imbu/gui
