# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
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

import logging



def get_logger(level=logging.INFO):
  level = level
  fmt = '%(asctime)s - %(message)s'
  datefmt = '%Y-%m-%d %H:%M:%S'
  formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

  _logger = logging.getLogger(__name__)
  _logger.setLevel(level)

  file_handler = logging.FileHandler('analysis.log')
  file_handler.setLevel(level)
  file_handler.setFormatter(formatter)
  _logger.addHandler(file_handler)

  console_handler = logging.StreamHandler()
  console_handler.setLevel(level)
  console_handler.setFormatter(formatter)
  _logger.addHandler(console_handler)
  return _logger



def check_shape(x, expected_shape):
  if x.shape != expected_shape:
    raise ValueError('Shape is %s but should be %s' % (x.shape, expected_shape))



def indent(indent_level, tick='.'):
  return '|' + '__' * indent_level + tick + ' '



def hours_minutes_seconds(timedelta):
  m, s = divmod(timedelta.seconds, 60)
  h, m = divmod(m, 60)
  return h, m, s
