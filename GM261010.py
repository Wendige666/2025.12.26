import gc
import os

import aicube
import image
import nncase_runtime as nn
import ujson
import ulab.numpy as np
from libs.PipeLine import ScopedTiming
from libs.Utils import *
from media.display import *
from media.media import *
from media.sensor import *

display_mode = "lcd"
if display_mode == "lcd":
    # ST7701 屏实际是 800x480；若画面被横向拉伸，把 640 改成 800
    DISPLAY_WIDTH = ALIGN_UP(640, 16)
    DISPLAY_HEIGHT = 480
else:
    DISPLAY_WIDTH = ALIGN_UP(1920, 16)
    DISPLAY_HEIGHT = 1080
OUT_RGB888P_WIDTH = ALIGN_UP(1280, 16)
OUT_RGB888P_HEIGH = 720

root_path = "/sdcard/mp_deployment_source/"
config_path = root_path + "deploy_config.json"
deploy_conf = {}
debug_mode = 1

# ==================== 识别区域 ROI ====================
ROI_ENABLE = True
ROI_W = 1280         # 识别区宽度（16 的倍数，<=1280）
ROI_H = 256          # 识别区高度（16 的倍数，<=720）
ROI_CENTER = True    # True=居中；False=用下面的绝对左上角
ROI_X = 240
ROI_Y = 232
ROI_COLOR = (255, 255, 0, 255)
ROI_THICKNESS = 3


def make_roi(src_w, src_h, w, h, center=True, x=0, y=0):
    """生成 ROI（左上角 x,y + 宽高），16 对齐并限制在原图内"""
    w = (int(min(w, src_w)) // 16) * 16
    h = (int(min(h, src_h)) // 16) * 16
    if center:
        x = (src_w - w) // 2
        y = (src_h - h) // 2
    x = (max(0, min(int(x), src_w - w)) // 16) * 16
    y = (max(0, min(int(y), src_h - h)) // 16) * 16
    return x, y, w, h


def two_side_pad_param(input_size, output_size):
    ratio_w = output_size[0] / input_size[0]
    ratio_h = output_size[1] / input_size[1]
    ratio = min(ratio_w, ratio_h)
    new_w = int(ratio * input_size[0])
    new_h = int(ratio * input_size[1])
    dw = (output_size[0] - new_w) / 2
    dh = (output_size[1] - new_h) / 2
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    return top, bottom, left, right, ratio


def read_deploy_config(config_path):
    with open(config_path, "r") as json_file:
        try:
            config = ujson.load(json_file)
        except ValueError as e:
            print("JSON 解析错误:", e)
    return config


def detection():
    print("det_infer start")
    deploy_conf = read_deploy_config(config_path)
    kmodel_name = deploy_conf["kmodel_path"]
    labels = deploy_conf["categories"]
    confidence_threshold = deploy_conf["confidence_threshold"]
    nms_threshold = deploy_conf["nms_threshold"]
    img_size = deploy_conf["img_size"]
    num_classes = deploy_conf["num_classes"]
    color_four = get_colors(num_classes)
    nms_option = deploy_conf["nms_option"]
    model_type = deploy_conf["model_type"]

    # ===== 修复 1：这些变量必须在 if 外面，否则非 AnchorBaseDet 时会 NameError =====
    if model_type == "AnchorBaseDet":
        anchors = deploy_conf["anchors"][0] + deploy_conf["anchors"][1] + deploy_conf["anchors"][2]
    kmodel_frame_size = img_size
    frame_size = [OUT_RGB888P_WIDTH, OUT_RGB888P_HEIGH]
    strides = [8, 16, 32]

    # ---- ROI ----
    if ROI_ENABLE:
        roi_x, roi_y, roi_w, roi_h = make_roi(
            OUT_RGB888P_WIDTH, OUT_RGB888P_HEIGH, ROI_W, ROI_H, ROI_CENTER, ROI_X, ROI_Y
        )
    else:
        roi_x, roi_y, roi_w, roi_h = 0, 0, OUT_RGB888P_WIDTH, OUT_RGB888P_HEIGH

    # 计算padding值：必须基于【裁剪后】的尺寸
    top, bottom, left, right, ratio = two_side_pad_param([roi_w, roi_h], kmodel_frame_size)

    # ===== 修复 2：下面这行旧代码必须删掉，它会把上面算对的值覆盖掉 =====
    # top, bottom, left, right, ratio = two_side_pad_param(frame_size, kmodel_frame_size)

    print("ROI(x,y,w,h)=", roi_x, roi_y, roi_w, roi_h,
          "| pad(t,b,l,r)=", top, bottom, left, right,
          "| scale=", round(ratio, 4))
    # 有效内容占模型输入的比例，太低说明 ROI 长宽比太极端，白喂了一堆灰边
    used_h = int(ratio * roi_h)
    used_w = int(ratio * roi_w)
    print("有效内容占比: %.1f%%" % (100.0 * used_h * used_w / (kmodel_frame_size[0] * kmodel_frame_size[1])))

    # 初始化kpu
    kpu = nn.kpu()
    kpu.load_kmodel(root_path + kmodel_name)

    # 初始化ai2d
    ai2d = nn.ai2d()
    ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
    if ROI_ENABLE:
        ai2d.set_crop_param(True, roi_x, roi_y, roi_w, roi_h)
    ai2d.set_pad_param(True, [0, 0, 0, 0, top, bottom, left, right], 0, [114, 114, 114])
    ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
    # build 输入仍是原始大图，crop 由硬件在大图上完成
    ai2d_builder = ai2d.build(
        [1, 3, OUT_RGB888P_HEIGH, OUT_RGB888P_WIDTH], [1, 3, kmodel_frame_size[1], kmodel_frame_size[0]]
    )

    # 初始化并配置sensor
    sensor = Sensor()
    sensor.reset()
    sensor.set_hmirror(False)
    sensor.set_vflip(False)
    sensor.set_framesize(width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
    sensor.set_pixformat(PIXEL_FORMAT_YUV_SEMIPLANAR_420)
    sensor.set_framesize(width=OUT_RGB888P_WIDTH, height=OUT_RGB888P_HEIGH, chn=CAM_CHN_ID_2)
    sensor.set_pixformat(PIXEL_FORMAT_RGB_888_PLANAR, chn=CAM_CHN_ID_2)

    sensor_bind_info = sensor.bind_info(x=0, y=0, chn=CAM_CHN_ID_0)
    Display.bind_layer(**sensor_bind_info, layer=Display.LAYER_VIDEO1)
    if display_mode == "lcd":
        Display.init(Display.ST7701, to_ide=True)
    else:
        Display.init(Display.LT9611, to_ide=True)

    osd_img = image.Image(DISPLAY_WIDTH, DISPLAY_HEIGHT, image.ARGB8888)

    MediaManager.init()
    sensor.run()

    # ROI 框在屏幕上的位置
    roi_dx = int(roi_x * DISPLAY_WIDTH // OUT_RGB888P_WIDTH)
    roi_dy = int(roi_y * DISPLAY_HEIGHT // OUT_RGB888P_HEIGH)
    roi_dw = int(roi_w * DISPLAY_WIDTH // OUT_RGB888P_WIDTH)
    roi_dh = int(roi_h * DISPLAY_HEIGHT // OUT_RGB888P_HEIGH)
    print("ROI on screen:", roi_dx, roi_dy, roi_dw, roi_dh)

    rgb888p_img = None
    ai2d_input_tensor = None
    data = np.ones((1, 3, kmodel_frame_size[1], kmodel_frame_size[0]), dtype=np.uint8)
    ai2d_output_tensor = nn.from_numpy(data)
    frame_cnt = 0

    while True:
        with ScopedTiming("total", debug_mode > 0):
            rgb888p_img = sensor.snapshot(chn=CAM_CHN_ID_2)
            if rgb888p_img.format() == image.RGBP888:
                ai2d_input = rgb888p_img.to_numpy_ref()
                ai2d_input_tensor = nn.from_numpy(ai2d_input)
                ai2d_builder.run(ai2d_input_tensor, ai2d_output_tensor)
                kpu.set_input_tensor(0, ai2d_output_tensor)
                kpu.run()

                results = []
                for i in range(kpu.outputs_size()):
                    out_data = kpu.get_output_tensor(i)
                    result = out_data.to_numpy()
                    result = result.reshape((result.shape[0] * result.shape[1] * result.shape[2] * result.shape[3]))
                    del out_data
                    results.append(result)

                # 第5个参数传模型尺寸 => 后处理不做坐标映射，我们自己反算
                det_boxes = aicube.anchorbasedet_post_process(
                    results[0],
                    results[1],
                    results[2],
                    kmodel_frame_size,
                    kmodel_frame_size,
                    strides,
                    num_classes,
                    confidence_threshold,
                    nms_threshold,
                    anchors,
                    nms_option,
                )

                osd_img.clear()
                # 画出识别区域
                if ROI_ENABLE:
                    osd_img.draw_rectangle(roi_dx, roi_dy, roi_dw, roi_dh,
                                           color=ROI_COLOR, thickness=ROI_THICKNESS)

                if det_boxes:
                    if frame_cnt % 20 == 0:
                        print("num:", len(det_boxes), "max_score:", round(max(b[1] for b in det_boxes), 3))
                    for det_boxe in det_boxes:
                        x1, y1, x2, y2 = det_boxe[2], det_boxe[3], det_boxe[4], det_boxe[5]
                        # 模型坐标 -> ROI 坐标 -> 原图坐标
                        x1 = (x1 - left) / ratio + roi_x
                        y1 = (y1 - top) / ratio + roi_y
                        x2 = (x2 - left) / ratio + roi_x
                        y2 = (y2 - top) / ratio + roi_y
                        # 夹回 ROI 内，避免画出界
                        x1 = max(roi_x, min(x1, roi_x + roi_w))
                        y1 = max(roi_y, min(y1, roi_y + roi_h))
                        x2 = max(roi_x, min(x2, roi_x + roi_w))
                        y2 = max(roi_y, min(y2, roi_y + roi_h))

                        x = int(x1 * DISPLAY_WIDTH // OUT_RGB888P_WIDTH)
                        y = int(y1 * DISPLAY_HEIGHT // OUT_RGB888P_HEIGH)
                        w = int((x2 - x1) * DISPLAY_WIDTH // OUT_RGB888P_WIDTH)
                        h = int((y2 - y1) * DISPLAY_HEIGHT // OUT_RGB888P_HEIGH)
                        osd_img.draw_rectangle(x, y, max(1, w), max(1, h), color=color_four[det_boxe[0]][1:])
                        text = labels[det_boxe[0]] + " " + str(round(det_boxe[1], 2))
                        osd_img.draw_string_advanced(x, max(0, y - 40), 32, text, color=color_four[det_boxe[0]][1:])
                else:
                    if frame_cnt % 20 == 0:
                        print("no box")

                Display.show_image(osd_img, 0, 0, Display.LAYER_OSD3)
                del ai2d_input_tensor
                del results
                gc.collect()
            rgb888p_img = None
            frame_cnt += 1

    del ai2d_input_tensor
    del ai2d_output_tensor
    sensor.stop()
    Display.deinit()
    MediaManager.deinit()
    gc.collect()
    time.sleep(1)
    nn.shrink_memory_pool()
    print("det_infer end")
    return 0


if __name__ == "__main__":
    detection()
