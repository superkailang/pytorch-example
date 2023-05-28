import importlib
import os

import gradio as gr
from omegaconf import OmegaConf

from annotator.util import resize_image, HWC3

config = OmegaConf.load("config/annotator.yaml")

package_annotator = "annotator"


def process_image(cls: str, img, res, *kwargs):
    img = resize_image(HWC3(img), res)
    # load_model()
    module_imp = importlib.import_module(package_annotator)
    model = getattr(module_imp, cls)
    image_processor = model()
    result = image_processor(img, *kwargs)
    if type(result) == tuple:
        return result
    return [result]


def process(cls):
    def process_fc(img, res, *args):
        return process_image(cls, img, res, *args)

    return process_fc


block = gr.Blocks().queue()
examples = [os.path.join(os.path.dirname(__file__), "examples/demo.jpeg")]
with block:
    for key in config.keys():
        cls, input_element = config[key]["process"], config[key].get("input")
        input_append = []
        with gr.Tab(key):
            with gr.Row():
                gr.Markdown("## " + key)
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(source='upload', type="numpy")
                    resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)

                    if input_element is not None:
                        for item in input_element:
                            input_append.append(getattr(gr, item["attr"])(**item["args"]))

                    run_button = gr.Button(label="Run")
                    gr.Examples(examples, input_image)
                with gr.Column():
                    gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")

            run_button.click(fn=process(cls),
                             inputs=[input_image, resolution] + input_append,
                             outputs=[gallery])

block.launch()
