# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

import shortuuid 
import sys
import matplotlib.animation as manimation

def trim_doc(docstring):
    """"
    Removes the indentation from a docstring.
    Thanks to: 
        http://codedump.tumblr.com/post/94712647/handling-python-docstring-indentation
    """
    if not docstring:
        return ''
    lines = docstring.expandtabs().splitlines()

    # Determine minimum indentation (first line doesn't count):
    indent =  sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))

    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())

    # Strip off trailing and leading blank lines:while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    return '\n'.join(trimmed)


def random_id(length):
    """Returns a random id of specified length."""
    return shortuuid.ShortUUID().random(length=length)


def create_movie(fig, update_figure, filename, title, fps=15, dpi=100):
    """Helps us to create a movie."""
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata     = dict(title=title)
    writer       = FFMpegWriter(fps=fps, metadata=metadata)

    with writer.saving(fig, filename, dpi):
        t = 0
        while True:
            if update_figure(t):
                writer.grab_frame()
                t += 1
            else:
                break


