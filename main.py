#!/usr/bin/env python3

import logging as log
import sys
from model_artifacts_downloader import ModelArtifactsDownloader
from ov_gpt2 import OVGPT2Config, OVGPT2
from argparse import ArgumentParser, SUPPRESS

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)


if __name__ == '__main__':

    ModelArtifactsDownloader().check_files('model')
    config = OVGPT2Config()
    gpt2 = OVGPT2(config)

    while True:
        x = input('Q:')
        response = gpt2.infer(x)
        print(f'GPT2: {response}\n')
        print("-" * 70)
