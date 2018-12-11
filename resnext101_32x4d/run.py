#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
LOG_FORMAT = "%(levelname)s:%(message)s"
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
BUILD_PATH = os.path.join(os.path.dirname(CUR_PATH), "build")
sys.path.insert(0, BUILD_PATH)

import timeit
import logging
import argparse
import numpy as np
import onnx
import cv2

from random import randint
from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import workspace as ws
from caffe2.python import transformations as tf

def CropCenter(img, cropx, cropy):
    import math
    y, x, c = img.shape
    startx = int(math.floor(x * 0.5 - (cropx * 0.5)))
    starty = int(math.floor(y * 0.5 - (cropy * 0.5)))

    imgCropped = img[starty : starty + cropy, startx : startx + cropx]
    logging.info("After cropped: {}".format(imgCropped.shape))
    return imgCropped

def Rescale(img, rescale_size):
    cv2_interpol = cv2.INTER_LINEAR
    logging.info("Original image shape: {} "
                 "and remember it should be in H, W, C!"
                 .format(str(img.shape)))
    logging.info("Model's input shape is {0} x {1}"
                 .format(rescale_size, rescale_size))
    aspect = img.shape[1] / float(img.shape[0])
    logging.info("Orginal aspect ratio: {}".format(str(aspect)))
    if aspect>=1:
        # landscape orientation - wide image
        res = int(rescale_size * aspect)
        imgScaled = cv2.resize(img, dsize=(res, rescale_size), interpolation=cv2_interpol)
    elif aspect<1:
        # portrait orientation - tall image
        res = int(rescale_size / aspect)
        imgScaled = cv2.resize(img, dsize=(rescale_size, res), interpolation=cv2_interpol)
    logging.info("After rescaled in HWC: {}".format(str(imgScaled.shape)))
    return imgScaled

def PreprocessSingleImage(image_path, crop_size, rescale_size, mean, scale, need_normalize, color_format):
    img = cv2.imread(image_path)
    img = img.astype(np.float32)
    img = Rescale(img, rescale_size)
    img = CropCenter(img, crop_size, crop_size)
    # switch to CHW
    img = img.swapaxes(1, 2).swapaxes(0, 1)
    # switch to RGB
    if color_format=='RGB':
        img = img[(2, 1, 0), :, :]

    if need_normalize==True:
        img=img/255
    if len(scale) == 1:
        img = (img - mean) * float(scale[0])
    elif len(scale) > 1:
        img[0, : ,:] = (img[0, :, :] - mean[0, :, :])*float(scale[0])
        img[1, : ,:] = (img[1, :, :] - mean[1, :, :])*float(scale[1])
        img[2, : ,:] = (img[2, :, :] - mean[2, :, :])*float(scale[2])
        logging.info("after img is {}".format(img))
    else:
        logging.error("scale = {} is invalid".format(scale))
        exit()
    # add batch size
    img = img[np.newaxis, :, :, :].astype(np.float32)
    logging.info("After Preprocessing in NCHW: {}".format(img.shape))
    return img

def PreprocessImages(img_paths, crop_size, rescale_size, mean, scale, need_normalize, color_format):
    imgs = []
    for i in img_paths:
        img = PreprocessSingleImage(i, crop_size, rescale_size, mean, scale, need_normalize, color_format)
        imgs.append(img)
    return np.concatenate(imgs, 0)

def UpdateDeviceOption(dev_opt, net_def):
    for eop in net_def.op:
        if (
                eop.device_option and
                eop.device_option.device_type != dev_opt.device_type
        ):
            eop.device_option.device_type = dev_opt.device_type

def OnnxToCaffe2(model_file):
    from caffe2.python.onnx import backend
    model = onnx.load(model_file)
    return backend.Caffe2Backend.onnx_graph_to_caffe2_net(model)

def LoadValidation(validation_file):
    if not os.path.isfile(validation_file):
        logging.error("Can not find validation file {}."
                      .format(validation_file))
        return None
    validation = {}
    with open(validation_file) as v:
        validation_lines = [line.rstrip('\n') for line in v.readlines()]
    for line in validation_lines:
        name, code = line.partition(" ")[::2]
        if name and code:
            name = name.strip()
            if name in validation:
                logging.warning("Repeated name {0} for code {1} in"
                                " validation file. Ignored!"
                                .format(name, code))
            else:
                validation[name] = int(code.strip())
    return validation

def BatchImages(images_path, batch_size, iterations):
    bs = batch_size
    images = []
    image = []
    fnames = []
    fname = []
    it=iterations
    for root, dirs, files in os.walk(images_path):
        if it== 0:
            break
        for fn in files:
            fp = os.path.join(root, fn)
            bs -= 1
            image.append(fp)
            fname.append(fn)
            if bs == 0:
                images.append(image)
                fnames.append(fname)
                image = []
                fname = []
                bs = batch_size
                it -= 1
                if it == 0:
                    break
    if len(image) > 0:
        images.append(image)
        fnames.append(fname)
    return (images, fnames)

def ParseOutputs(outputs):
    total = 0
    parsed_outputs = []
    for i, output in enumerate(outputs):
        for j, o in enumerate(np.split(output, output.shape[0], 0)):
            total += 1
            o = np.squeeze(o)
            index = []
            score = []
            z = 0
            while z < 5:
                z += 1
                index.append(np.argmax(o))
                score.append(o[np.argmax(o)])
                o[np.argmax(o)] = -1
            logging.info("The index is {}".format(index))
            logging.info("The score is {}".format(score))
            parsed_outputs.append([index, score, (i, j)])
    return (parsed_outputs, total)

def ParseResults(results, validation, fnames):
    summary = []
    for result in results:
        index = result[0]
        highest = result[1][0]
        file_pos = result[2]
        fname = fnames[file_pos[0]][file_pos[1]]
        if fname in validation:
            if validation[fname] == index[0]:
                logging.info("Validation passed for file {0} index[0]"
                             " {1} with a {2:.5%} probability."
                             .format(fname, index[0], highest))
                summary.append((fname, "Pass", index[0], index[0], highest))
            elif validation[fname] in index:
                logging.info("Validation partially passed for file {0} index[0]"
                             " {1} with a {2:.5%} top1 probability."
                             .format(fname, validation[fname], highest))
                summary.append((fname, "Top5Pass", validation[fname], index[0], highest))
            else:
                logging.info("Failed in validation for file {0} index"
                             " {1}. Should be {2}."
                             .format(fname, index[0], validation[fname]))
                summary.append(
                    (fname, "Fail", index[0], validation[fname], highest))
        else:
            logging.error("Can NOT find the file {} in validation!"
                          .format(fname))
    return summary

def Run(args, extra_args):
    logging.warning("Run Caffe2 in inference mode with args:\n{}".format(vars(args)))
    images_path = os.path.abspath(args.images_path)
    validation = LoadValidation(args.validation_file)
    iterations = args.iterations if args.iterations else sys.maxsize
    warmup_iter = args.warmup_iterations if args.warmup_iterations > 0 else 0
    batch_size = int(args.batch_size)

    model_name = "ReNeXt101_32x4d"
    model_onnx = os.path.join(CUR_PATH, "resnext101_32x4d.onnx")
    crop_size = 224
    rescale_size=256
    color_format="RGB"
    need_normalize = True
    scale=[4.3668, 4.4643, 4.4444]
    mean_tmp = [0.485, 0.456, 0.406]
    mean = np.zeros([3, crop_size, crop_size], dtype=np.float)
    mean[0, :, :] = float(mean_tmp[0])  # 104
    mean[1, :, :] = float(mean_tmp[1])  # 117
    mean[2, :, :] = float(mean_tmp[2])  # 124

    dev_map = {
        "cpu": caffe2_pb2.CPU,
        "ideep": caffe2_pb2.IDEEP,
    }
    device_opts_cpu = caffe2_pb2.DeviceOption()
    device_opts_cpu.device_type = caffe2_pb2.CPU
    device_opts = caffe2_pb2.DeviceOption()
    if  args.device.lower() in dev_map:
        device_opts.device_type = dev_map[args.device.lower()]
    else:
        logging.error("Wrong device {}. Exit!".format(args.device))
        return

    init_def, predict_def = OnnxToCaffe2(model_onnx)
    UpdateDeviceOption(device_opts, predict_def)
    UpdateDeviceOption(device_opts, init_def)
    init_label = np.ones((batch_size), dtype=np.int32)
    init_data = np.random.rand(batch_size, 3, crop_size, crop_size).astype(np.float32)
    ws.FeedBlob(str(predict_def.op[0].input[0]), init_data, device_opts)
    ws.RunNetOnce(init_def)

    net = core.Net(model_name)
    net.Proto().CopyFrom(predict_def)
    if args.device.lower() == 'ideep':
        logging.warning('optimize....................')
        tf.optimizeForIDEEP(net)

    predict_def = net.Proto()
    if predict_def.op[-1].type == 'Accuracy':
        label = net.AddExternalInput('label')
        ws.FeedBlob(label, init_label, device_opts_cpu)
        for i, op in enumerate(predict_def.op):
            if op.type == 'Accuracy':
                ws.FeedBlob(str(predict_def.op[i].output[0]), init_label, device_opts_cpu)
    ws.CreateNet(net)

    images=[]
    fnames=[]
    images, fnames = BatchImages(images_path, batch_size, iterations)
    logging.warning("Start warmup {} iterations...".format(warmup_iter))
    wi=warmup_iter-1
    while warmup_iter:
        warmup_iter -= 1
        r = randint(0, len(images) - 1)
        imgs = PreprocessImages(images[r], crop_size, rescale_size, mean, scale, need_normalize, color_format)
        ws.FeedBlob(str(predict_def.op[0].input[0]), imgs, device_opts)
        ws.RunNet(net)

    comp_time = 0
    processed_images = 0
    outputs=[]
    accuracy_top1 = []
    accuracy_top5 = []
    logging.warning("Start running net {}".format(model_name))
    for k, raw in enumerate(images):
        processed_images += len(raw)
        imgs = PreprocessImages(raw, crop_size, rescale_size, mean, scale, need_normalize, color_format)
        init_label = None
        if predict_def.op[-1].type == 'Accuracy' and len(validation) > 0:
            batch_fname = fnames[k]
            init_label = np.ones((len(fnames[k])), dtype=np.int32)
            for j in range(len(fnames[k])):
                init_label[j] = validation[batch_fname[j]]

        ws.FeedBlob(str(predict_def.op[0].input[0]), imgs, device_opts)
        if predict_def.op[-1].type == 'Accuracy':
            ws.FeedBlob(str(predict_def.op[-1].input[1]), init_label, device_opts_cpu)
            ws.FeedBlob(str(predict_def.op[-2].input[1]), init_label, device_opts_cpu)

        comp_start_time = timeit.default_timer()
        ws.RunNet(net)
        comp_elapsed_time = timeit.default_timer() - comp_start_time
        comp_time += comp_elapsed_time

        output = ws.FetchBlob(str(predict_def.op[-1].output[0]))
        if predict_def.op[-1].type == 'Accuracy':
            accuracy_top5.append(output)
            accuracy_top1.append(ws.FetchBlob(str(predict_def.op[-2].output[0])))
        else:
            outputs.append(output)
        logging.warning("[{0:.2%}] Output shape: {1}, computing in {2:.10f}"
                        " seconds, processing {3} images."
                        .format(((k + 1) / len(images)), output.shape,
                                comp_elapsed_time, len(raw)))

        del imgs
        if k >= (iterations - 1):
            logging.warning("Exit after running {} iterations"
                            .format(iterations))
            break

    if comp_time <= 0:
        logging.error("The total time is invalid!")
        return

    info_str = ""
    if predict_def.op[-1].type == 'Accuracy':
        mean_accuracy_top1 = 0
        mean_accuracy_top5 = 0
        for i in range(len(accuracy_top1)):
            mean_accuracy_top1 += accuracy_top1[i] * batch_size;
            mean_accuracy_top5 += accuracy_top5[i] * batch_size;
        mean_accuracy_top1 /= (batch_size * len(accuracy_top1))
        mean_accuracy_top5 /= (batch_size * len(accuracy_top5))
        info_str = "\nAccuracy: {:.5%}".format(mean_accuracy_top1)
        info_str += "\nTop5Accuracy: {:.5%}".format(mean_accuracy_top5)
        total_image = processed_images
        logging.critical("\nImages per second: {0:.10f}\nTotal computing time:"
                         " {1:.10f} seconds\nTotal images: {2}{3}"
                         .format(total_image / comp_time, comp_time,
                             total_image, info_str))
    else:
        results, total_image = ParseOutputs(outputs)
        summary = ParseResults(results, validation, fnames)
        if not summary:
            logging.error("Failed to parse the results!")
            return
        elif total_image <= 0 or len(summary) != total_image:
            logging.error("No available results!")
            return

        accuracy = 0
        top5accuracy = 0
        for res in summary:
            if res[1] == "Pass":
                accuracy += 1
                top5accuracy += 1
            elif res[1] == "Top5Pass":
                top5accuracy += 1

        accuracy = accuracy / total_image
        top5accuracy = top5accuracy / total_image
        info_str += "\nAccuracy: {:.5%}".format(accuracy)
        info_str += "\nTop5Accuracy: {:.5%}".format(top5accuracy)

        logging.critical("\nImages per second: {0:.10f}\nTotal computing time:"
                     " {1:.10f} seconds\nTotal images: {2}{3}"
                     .format(total_image / comp_time, comp_time,
                         total_image, info_str))


def GetArgumentParser():
    parser = argparse.ArgumentParser(description="The scripts to run Caffe2.\n")
    parser.add_argument(
        "-p", "--images_path",
        type=str,
        default=os.path.join(CUR_PATH, "images"),
        help="The path of input images. (DEFAULT: %(default)s)"
    )
    parser.add_argument(
        "-v", "--validation_file",
        type=str,
        default=os.path.join(CUR_PATH, "val.txt"),
        help="The input validation index file. (DEFAULT: %(default)s)"
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=1,
        help="The batch size. (DEFAULT: %(default)i)"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="ideep",
        help="Choose device to run. cpu, gpu or ideep."
             "(DEFAULT: %(default)s)"
    )
    parser.add_argument(
        "-i", "--iterations",
        type=int,
        help="Number of iterations to run the network. (DEFAULT: None)"
    )
    parser.add_argument(
        "-w", "--warmup_iterations",
        type=int,
        default=0,
        help="Number of warm-up iterations before benchmarking."
             "(DEFAULT: %(default)i)"
    )
    return parser

if __name__ == '__main__':
    args, extra_args = GetArgumentParser().parse_known_args()
    logging.basicConfig(
        format=LOG_FORMAT,
        filename=None,
        filemode="w",
        level=logging.WARNING)

    if args.images_path is None or args.validation_file is None:
        GetArgumentParser().print_help()
        exit()

    Run(args, extra_args)

