# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import sys
import platform
import distutils.sysconfig
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class PreBuiltExt(build_ext):
    def build_extension(self, ext):
        print("Using prebuilt extension library")
        libdir="lib.%s-%s-%s" % (sys.platform, platform.machine(), distutils.sysconfig.get_python_version())
        os.system("mkdir -p build/%s" % libdir)
        os.system("cp -p vespa_ann_bm.*.so build/%s" % libdir)

setup(
  name="vespa_ann_bm",
  version="0.0.0",
  author="Tor Egge",
  author_email="Tor.Egge@online.no",
  description="Python binding for vespa ann benchmark fixture using tensor attribute",
  long_description="Python binding for vespa ann benchmark fixture using tensor attribute -- long version",
  ext_modules=[CMakeExtension("vespa_ann_bm")],
  cmdclass={"build_ext": PreBuiltExt},
  zip_safe=False,
)
