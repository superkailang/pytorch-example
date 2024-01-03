import yaml
from omegaconf import OmegaConf
import json
import os
from annotator.util import resize_image, HWC3

config = {
    "canny": {
        "process": "CannyDetector",
        "input": [
            {
                "attr": "Slider",
                "args": {
                    "label": "low_threshold",
                    "minimum": 1,
                    "maximum": 255,
                    "value": 100,
                    "step": 1
                }
            }, {
                "attr": "Slider",
                "args": {
                    "label": "high_threshold",
                    "minimum": 1,
                    "maximum": 255,
                    "value": 200,
                    "step": 1
                }
            }
        ]
    },
    "hed": {
        "process": "HedDetector",
        "input": []
    },
    "mlsd": {
        "process": "MLSDProcessor",
        "input": [
            {
                "attr": "Slider",
                "args": {
                    "label": "value_threshold",
                    "minimum": 0.01,
                    "maximum": 2.0,
                    "value": 0.1,
                    "step": 0.01
                }
            },
            {
                "attr": "Slider",
                "args": {
                    "label": "distance_threshold",
                    "minimum": 0.01,
                    "maximum": 20.0,
                    "value": 0.1,
                    "step": 0.01
                }
            }
        ]
    },
    "midas": {
        "process": "MidasProcessor",
        "input": [
            {
                "attr": "Slider",
                "args": {
                    "label": "alpha",
                    "minimum": 0.1,
                    "maximum": 20,
                    "value": 6.2,
                    "step": 0.01
                }
            }
        ]
    },
    "openpose": {
        "process": "OpenposeDetector",
        "input": [
            {
                "attr": "Checkbox",
                "args": {
                    "label": "detect hand",
                    "value": False
                }
            }
        ]
    },
    "uniformer": {
        "process": "UniformerDetector",
        "input": []
    },
    "lineArt": {
        "process": "LineArtDetector",
        "input": [],
    },
    "lineArtAnime": {
        "process": "LineArtAnimeDetector",
        "input": [],
    },
    "mangaLine": {
        "process": "MangaLineExtration",
    },
    "normalBae": {
        "process": "NormalBaeDetector",
    },
    "leres": {
        "process": "LeresPix2Pix",
        "input": [
            {
                "attr": "Slider",
                "args": {
                    "label": "thr_a",
                    "minimum": 0,
                    "maximum": 250,
                    "value": 100,
                    "step": 1
                }
            },
            {
                "attr": "Slider",
                "args": {
                    "label": "thr_b",
                    "minimum": 0,
                    "maximum": 250,
                    "value": 200,
                    "step": 1
                }
            },
            {
                "attr": "Checkbox",
                "args": {
                    "label": "boost",
                    "value": False
                }
            }
        ]
    },
    "meidaPipe": {
        "process": "MediaPipeFace",
        "input": [
            {
                "attr": "Slider",
                "args": {
                    "label": "max_faces",
                    "minimum": 1,
                    "maximum": 100,
                    "value": 1,
                    "step": 1
                }
            },
            {
                "attr": "Slider",
                "args": {
                    "choices": ["USA", "Japan", "Pakistan"],
                    "label": "min_confidence",
                    "minimum": 0,
                    "maximum": 1,
                    "value": 0.5,
                    "step": 0.1
                }
            }
        ]
    }
}


# 原json文件同级目录下，生成yaml文件
# 生成文件
def generate_file(filePath, datas):
    if os.path.exists(filePath):
        os.remove(filePath)
    with open(filePath, 'w') as f:
        f.write(datas)


# json文件内容转换成yaml格式
def json_to_yaml(datas):
    # with open(json_path, encoding="utf-8") as f:
    #     datas = json.load(f)  # 将文件的内容转换为字典形式
    yaml_datas = yaml.dump(datas, indent=5, sort_keys=False, allow_unicode=True)  # 将字典的内容转换为yaml格式的字符串
    return yaml_datas


generate_file("../config/annotator2.yaml", json_to_yaml(config))
