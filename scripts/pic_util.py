from tagger.tagger import utils
import os.path
import time

import cv2
import numpy
from PIL import Image, ImageOps

import subprocess

import modules.images
import os
import sys
from modules.shared import opts, state
from modules import shared, sd_samplers, processing, images, scripts_postprocessing
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img, process_images, Processed
import rembg
from scripts.m2a_config import pic_output_dir

import time
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor

_kernel = None

def upscale(image, info, upscaler, upscale_mode=1, upscale_by=2, upscale_to_width=1080, upscale_to_height=1920, upscale_crop=False):
    if upscale_mode == 1:
        upscale_by = max(upscale_to_width / image.width, upscale_to_height / image.height)
        info["Postprocess upscale to"] = f"{upscale_to_width}x{upscale_to_height}"
    else:
        info["Postprocess upscale by"] = upscale_by
    image = upscaler.scaler.upscale(image, upscale_by, upscaler.data_path)

    if upscale_mode == 1 and upscale_crop:
        cropped = Image.new("RGB", (upscale_to_width, upscale_to_height))
        cropped.paste(image, box=(upscale_to_width // 2 - image.width // 2, upscale_to_height // 2 - image.height // 2))
        image = cropped
        info["Postprocess crop to"] = f"{image.width}x{image.height}"

    return image

def run_upscale(image, upscale_to_width=1080, upscale_to_height=1920, upscaler_name='R-ESRGAN 4x+'):
    if upscaler_name == 'None':
        upscaler_name = 'R-ESRGAN 4x+'
    info = {}
    upscaler = next(iter([x for x in shared.sd_upscalers if x.name == upscaler_name]), None)
    return upscale(image, info, upscaler, 1, 1, upscale_to_width, upscale_to_height)


def refresh_interrogators():
    if not bool(utils.interrogators):
        utils.refresh_interrogators()
    print('utils.interrogators', utils.interrogators)

def getTagsFromImage(image, isImage, invoke_tagger_val, common_invoke_tagger):
    key = 'wd14-vit-v2-git'
    interrogator = utils.interrogators[key]
    img_pil = image
    if not isImage:
        img_pil = Image.fromarray(image)
    print('img_pil', img_pil)
    ratings, tags = interrogator.interrogate(img_pil)

    tagsSelect = []

    for k in tags:
        tagV = tags[k]
        if tagV > invoke_tagger_val and k != 'realistic':
            tagsSelect.append(k)

    # ret = '((masterpiece)),best quality,ultra-detailed,smooth,best illustrationbest, shadow,photorealistic,hyperrealistic,backlighting,' + ','.join(tagsSelect)
    ret = common_invoke_tagger + ','.join(tagsSelect)
    return ret

# p 图生图或者文生图实例
# originPics 要转换的图片数组
# invoke_tagger 开启反推提示词
# invoke_tagger_val 反推提示词阈值
# common_invoke_tagger 公共提示词
def process_pic(p, originPics, invoke_tagger, invoke_tagger_val, common_invoke_tagger, des_enabled,des_width,des_height,upscaler_name):
    print('originPics:', originPics)
    images = originPics
    if not images:
        print('Failed to parse the video, please check')
        return
    print(f'The video conversion is completed, images:{len(images)}')
    max_frames = len(images)

    p.do_not_save_grid = True
    state.job_count = max_frames

    generate_images = []
    if invoke_tagger:
        refresh_interrogators()

    if not os.path.exists(pic_output_dir):
        os.makedirs(pic_output_dir, exist_ok=True)

    now = time.time()
    now = int(now)
    outDir = os.path.join(pic_output_dir, str(now))

    for i, image in enumerate(images):
        if i >= max_frames:
            break
        state.job = f"{i + 1} out of {max_frames}"
        if state.skipped:
            state.skipped = False

        if state.interrupted:
            break

        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 'RGB')
        img = ImageOps.exif_transpose(img)

        # 修改prompt
        if invoke_tagger:
            newTag = getTagsFromImage(img, True, invoke_tagger_val, common_invoke_tagger)
            p.prompt = newTag
            print('p.prompt 改为：', newTag)
        p.init_images = [image]

        print(f'current progress: {i + 1}/{max_frames}')
        processed = process_images(p)
        # 只取第一张

        gen_image = processed.images[0]
        # 判断是否需要放缩
        if des_enabled:
            gen_image = run_upscale(gen_image, des_width, des_height, upscaler_name)


        print('这是第',i,'张图片:', gen_image)



        tmpimage = images.save_image(gen_image, outDir, "", p.seed, p.prompt,
                                         forced_filename=str(i))
        generate_images.append(tmpimage)

    return generate_images