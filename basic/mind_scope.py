import PIL
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-to-speech', 'damo/speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k')

# p = pipeline('universal-matting', 'damo/cv_unet_universal-matting')


import patoolib

patoolib.extract_archive(file_path, outdir="")


PIL.Image.open()