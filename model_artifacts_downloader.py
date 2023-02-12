import subprocess
from pathlib import Path
import hashlib
from urllib.parse import urlparse
import shutil
from tqdm import tqdm
import requests
import logging as log
from openvino.runtime import serialize
from openvino.tools import mo
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.onnx import export, FeaturesManager


class ModelArtifactsDownloader:
    def __init__(self):
        pass

    def check_file(self, path: str, file: str, checksum: str) -> bool:
        full_path = Path(path) / Path(file)
        if full_path.is_file():
            sha_hash = hashlib.sha384()
            with open(full_path, "rb") as f:
                # Read and update hash string value in blocks of 4K
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha_hash.update(byte_block)
                computed_checksum = sha_hash.hexdigest()
                if checksum == computed_checksum:
                    return True
        return False

    def download_file(self, path: str, url: str) -> str:
        parsed = urlparse(url)
        full_path = Path(path) / Path(parsed.path).name
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(full_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            log.error("ERROR, in download. Expected {} bytes but downloaded {} bytes"
                      .format(total_size_in_bytes, progress_bar.n))
        else:
            return full_path

    def check_files(self, path: str):
        model_name = 'gpt-2'
        base_model_dir = 'model_converter'
        cache_dir = Path(base_model_dir) / 'cache'
        precision = 'FP16'

        if not Path(path + '/' + model_name).with_suffix('.xml').exists():
            # download_command = [f'omz_downloader',
            #                     f'--name {model_name}',
            #                     f'--output_dir {base_model_dir}',
            #                     f'--cache_dir {cache_dir}']
            # log.info(f"Download command: `{download_command}`")
            # log.info(f"Downloading {model_name}... (This may take a few minutes depending on your connection.)")
            # subprocess.run(download_command)

            # Convert & Rename layers
            pt_model = GPT2LMHeadModel.from_pretrained('gpt2')
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

            # define path for saving onnx model
            onnx_path = Path('model/gpt-2.onnx')
            onnx_path.parent.mkdir(exist_ok=True)

            # get model onnx config function for output feature format casual-lm
            model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(pt_model, feature='causal-lm')

            # fill onnx config based on pytorch model config
            onnx_config = model_onnx_config(pt_model.config)

            # convert model to onnx
            onnx_inputs, onnx_outputs = export(tokenizer, pt_model, onnx_config, onnx_config.default_onnx_opset,
                                               onnx_path)

            # convert model to openvino
            ov_model = mo.convert_model(onnx_path, compress_to_fp16=True,
                                        input='input_ids[1,1..128],attention_mask[1,1..128]')

            # serialize openvino model
            serialize(ov_model, str(Path(path + '/' + model_name).with_suffix('.xml')))

        # Path(path).mkdir(exist_ok=True)
        # if not Path(Path(path) / Path(model_name + '.bin')).exists():
        #     shutil.copy(converted_model_path, Path(path))
        #
        # if not Path(Path(path) / Path(model_name + '.xml')).exists():
        #     shutil.copy(converted_xml_path, Path(path))
        #
        # if not Path(Path(path) / Path('vocab.json')).exists():
        #     shutil.copy(vocab_path, Path(path))
        #
        # if not Path(Path(path) / Path('merges.txt')).exists():
        #     shutil.copy(merges_path, Path(path))



    def _check_files(self):
        # ONNX model from:
        # https://github.com/onnx/models/tree/main/text/machine_comprehension/gpt-2/model
        # vocab & merges from:
        # https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/gpt-2/model.yml

        artifacts = [
            {
                'file': 'https://media.githubusercontent.com/media/onnx/models/main/text/machine_comprehension/gpt-2/model/gpt2-10.onnx',
                'checksum': 'd27567c2838f21024006248d21ccb3817e9897113e6485edab8b8b24aef404c681559f07307d35a2c63e8eb33a8e582a'
            },
            {
                'file': 'https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/gpt-2/vocab.json',
                'checksum': '43e578a41ade90c90c71fdc4bfc1457c2e288c0450b872f3fb4a27e9521cadcf881aa6f24e5fae208277d4134991c7b0'
            },
            {
                'file': 'https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/gpt-2/merges.txt',
                'checksum': 'f91aa09e8551d2e001b4d6d0bf9a350a194ffe19397b913094c5ad8da940093894fd4064fce8edde9792b3474c107038'
            },
        ]

        for artifact in artifacts:
            parsed = urlparse(artifact['file'])
            if not self.check_file('.', Path(parsed.path).name, artifact['checksum']):
                log.info("Downloading: {}".format(artifact['file']))
                self.download_file('.', artifact['file'])
                if not self.check_file('.', Path(parsed.path).name, artifact['checksum']):
                    raise RuntimeError('checksum for downloaded file {} failed'.format(Path(parsed.path).name))
                else:
                    log.info('File {} downloaded successfully'.format(Path(parsed.path).name))
            else:
                log.debug('File {} checksum looks good'.format(Path(parsed.path).name))
