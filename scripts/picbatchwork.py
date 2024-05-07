import gradio as gr
from pathlib import Path
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from modules import script_callbacks, shared, scripts
from modules.ui_components import ToolButton, FormRow, FormGroup
from modules.ui_common import folder_symbol, plaintext_to_html
import json
import os
import shutil
import sys
import platform
import subprocess as sp
from modules.shared import opts, state

from scripts.m2a_config import pic_output_dir
# from scripts.xyz import init_xyz

from scripts.pic_util import process_pic

NAME = 'picBatchWork'

print(NAME)




def open_folder(f):
    print('打开文件夹：', pic_output_dir)
    if not os.path.exists(f):
        print(f'Folder "{f}" does not exist. After you create an image, the folder will be created.')
        return
    elif not os.path.isdir(f):
        print(f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
""", file=sys.stderr)
        return

    if not shared.cmd_opts.hide_ui_dir_config:
        path = os.path.normpath(f)
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            sp.Popen(["open", path])
        elif "microsoft-standard-WSL2" in platform.uname().release:
            sp.Popen(["wsl-open", path])
        else:
            sp.Popen(["xdg-open", path])

class Script(scripts.Script):
    def __init__(self):
        super().__init__()
        self.picBatchWorkScriptIsRuning = False

    def title(self):
        return NAME

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(NAME, open=False):
            enabled = gr.Checkbox(label='Enabled', value=False)
            with FormRow().style(equal_height=False):
                with gr.Column(variant='compact', elem_id=f"m2a_settings"):
                    init_pic_dir = gr.Textbox(label="批量转换目录", elem_id="m2a_pic_dir", show_label=True, lines=1,
                                              placeholder="请输入批量图片目录")
            with FormRow():
                des_enabled = gr.Checkbox(label='启用放缩至指定宽高', value=False)

            with FormRow():
                des_width = gr.Number(label='目标宽度', value=1080,
                                            elem_id='des_width')
                des_height = gr.Number(label='目标高度', value=1920,
                                    elem_id='des_height')
                upscaler_name = gr.Dropdown(label='放大算法', elem_id="extras_upscaler_1",
                                                choices=[x.name for x in shared.sd_upscalers],
                                                value=shared.sd_upscalers[0].name)
            with FormRow():
                invoke_tagger = gr.Checkbox(label='是否启用反推提示词', value=False)
            with FormRow():
                invoke_tagger_val = gr.Number(label='反推提示词阈值', value=0.35,
                                            elem_id='m2a_invoke_tag_val')
            with FormRow():
                common_invoke_tagger = gr.Textbox(label="如果你使用反推提示词，请输入你想附加的正向tag", elem_id="m2a_common_invoke_tagger", show_label=True, lines=3,
                                              placeholder="通用正向提示词，例如：((masterpiece)),best quality,ultra-detailed,smooth,best illustrationbest, shadow,photorealistic,hyperrealistic,backlighting,")
            with FormRow():
                open_folder_button = gr.Button(folder_symbol,
                                               elem_id="打开生成目录")

            open_folder_button.click(
                fn=lambda: open_folder(pic_output_dir),
                inputs=[],
                outputs=[],
            )
        return [
            enabled,
            init_pic_dir,
            invoke_tagger,
            invoke_tagger_val,
            common_invoke_tagger,
            des_enabled,
            des_width,
            des_height,
            upscaler_name,
        ]

    def invokePicWork(
            self,
            p: StableDiffusionProcessing,
            init_pic_dir: str,
            invoke_tagger: bool,
            invoke_tagger_val: int,
            common_invoke_tagger: str,
            des_enabled: bool,
            des_width: int,
            des_height: int,
            upscaler_name: str,
    ):
        originPics = []

        m_files = os.listdir(init_pic_dir)
        for file_name in m_files:
            m_file = os.path.join(init_pic_dir, file_name)
            originPics.append(m_file)

        desPics = process_pic(p, originPics,
                            invoke_tagger, invoke_tagger_val, common_invoke_tagger, des_enabled,des_width,des_height,upscaler_name)

        for pic in desPics:
            print('pic batch work complete, output file is ', pic)

    def process(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        init_pic_dir: str,
        invoke_tagger: bool,
        invoke_tagger_val: int,
        common_invoke_tagger: str,
        des_enabled: bool,
        des_width: int,
        des_height: int,
        upscaler_name: str,
    ):
        if enabled and not self.picBatchWorkScriptIsRuning:
            try:
                self.picBatchWorkScriptIsRuning = True
                if not init_pic_dir:
                    raise Exception('Error！ Please add a video file!')

                self.invokePicWork(
                    p,
                    init_pic_dir,
                    invoke_tagger,
                    invoke_tagger_val,
                    common_invoke_tagger,
                    des_enabled,
                    des_width,
                    des_height,
                    upscaler_name,
                )
            finally:
                self.picBatchWorkScriptIsRuning = False
                state.interrupted = True