#!/usr/bin/env python3

import logging as log
import sys
from model_artifacts_downloader import ModelArtifactsDownloader
from ov_gpt2 import OVGPT2Config, OVGPT2
from argparse import ArgumentParser, SUPPRESS

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-i", "--input", help="Optional. Input prompt", required=False, type=str, action='append')
    return parser


if __name__ == '__main__':

    parser = build_argparser().parse_args()

    ModelArtifactsDownloader().check_files('model')
    config = OVGPT2Config()
    gpt2 = OVGPT2(config)

    if parser.input:
        def prompts():
            for prompt in parser.input:
                log.info("Q: {}".format(prompt))
                yield prompt
    else:
        def prompts():
            while True:
                yield input('Q:')

    for prompt in prompts():
        response = gpt2.infer(prompt)
        print(f'GPT2: {response}\n')
        print("-" * 70)
