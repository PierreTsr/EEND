#!/bin/bash

wget https://us.openslr.org/resources/12/dev-clean.tar.gz
wget https://us.openslr.org/resources/12/test-clean.tar.gz
wget https://us.openslr.org/resources/12/train-clean-100.tar.gz
wget https://us.openslr.org/resources/12/train-clean-360.tar.gz

tar -xkf dev-clean.tar.gz
tar -xkf test-clean.tar.gz
tar -xkf train-clean-100.tar.gz
tar -xkf train-clean-360.tar.gz