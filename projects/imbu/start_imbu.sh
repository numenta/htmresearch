#!/bin/bash
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
if [ -f /opt/numenta/nupic.research/setup.py ]; then
  pushd /opt/numenta/nupic.research
  # Remove previously installed htmresearch in favor of local clone
  pip uninstall --yes htmresearch
  # Install local htmresearch in development/editable mode
  python setup.py develop
  popd
fi
# Re-install in development mode
python setup.py develop
mkdir -p logs
sudo nginx -p . -c conf/nginx-fluent.conf
supervisord -c conf/supervisord.conf --nodaemon
