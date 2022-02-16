#!/usr/bin/env python

import setuptools
import site
import sys

# https://github.com/googlefonts/fontmake/commit/164b24fd57c062297bf68b21c8ae88bc676f090b
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

if __name__ == "__main__":
    setuptools.setup()
