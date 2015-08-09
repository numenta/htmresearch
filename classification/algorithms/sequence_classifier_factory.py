# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""Module providing a factory for instantiating a Sequence classifier."""

from SequenceClassifier import SequenceClassifier
from nupic.support.configuration import Configuration



class SequenceClassifierFactory(object):
  """Factory for instantiating Sequence classifiers."""


  @staticmethod
  def create(*args, **kwargs):
    impl = kwargs.pop('implementation', None)
    if impl is None:
      #TODO: update that. Using old CLA impl for now.
      impl = Configuration.get('nupic.opf.claClassifier.implementation')
    if impl == 'py':
      return SequenceClassifier(*args, **kwargs)
    elif impl == 'cpp':
      raise ValueError('cpp version not yet implemented')
    else:
      raise ValueError('Invalid classifier implementation (%r). Value must be '
                       '"py" or "cpp".' % impl)
