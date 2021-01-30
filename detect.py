import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from ExtraUtils import *
from detection_heatmap import *

def detect(save_img=False):
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = False
        view_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colors = get_fix_colors(len(names))

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once

    # TODO cater for multiple files detection
    # save the prev pred res
    prev_pred = None
    frame_rate = 5
    prev_pred_time = 0
    cut_frame = True
    # save the exposure info for the heatmap
    prev_dash_time = 0
    current_exposure = None
    final_exposure = None

    for path, img, im0s, vid_cap, vid_dur in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        process_pred = True

        # Inference
        t1 = torch_utils.time_synchronized()

        if cut_frame:
            #if input is not video or prev pred is none or 1 per frame rate
            time_elapsed = torch_utils.time_synchronized() - prev_pred_time
            if vid_cap is None or prev_pred is None or time_elapsed > 1. / frame_rate:
                prev_pred_time = torch_utils.time_synchronized()
                pred = model(img, augment=opt.augment)[0]
                prev_pred = pred
            else:
                pred = prev_pred
                process_pred = False
        else:
            pred = model(img, augment=opt.augment)[0]

        t2 = torch_utils.time_synchronized()

        if process_pred:
            # to float
            if half:
                pred = pred.float()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                       multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections for image i
                # print('Frame start:::::::::::::::::::::::::::::::::::::::::::::::')
                # print('det result:', type(det))
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from imgsz to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        if int(c) > len(names)-1:
                            print('-------------------------------------------------- weird c', img.name)
                            s += '%g weird c = %s, ' % (n, int(c))
                        else:
                            s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    detected_xyxy = []

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # detected_xyxy.append(np.array([ [int(xyxy[0]), int(xyxy[1])], [int(xyxy[0]), int(xyxy[3])],
                        #                       [int(xyxy[2]), int(xyxy[3])], [int(xyxy[2]), int(xyxy[1])] ], np.int32))
                        detected_xyxy.append(np.array([ [xyxy[0], xyxy[1]], [xyxy[0], xyxy[3]],
                                                       [xyxy[2], xyxy[3]], [xyxy[2], xyxy[1]] ], np.int32))

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                                file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        if save_img or view_img:  # Add bbox to image
                            if int(cls) > len(names)-1:
                                label = '%s %.2f' % ('weird c', conf)
                                colour = colors[len(names)-1]
                            else:
                                label = '%s %.2f' % (names[int(cls)], conf)
                                colour = colors[int(cls)]
                            plot_one_box(xyxy, im0, label=label, color=colour)
                            # draw_grid(im0)

                current_dash_time = torch_utils.time_synchronized()
                new_exposures = accumulate_exposures(im0.shape, detected_xyxy, 1./frame_rate)

                # accumulated heatmap
                if final_exposure is None:
                    final_exposure = np.zeros((im0.shape[0], im0.shape[1]), dtype=np.float)
                final_exposure = final_exposure + new_exposures

                norm_exp = vid_heatmap_normalize(final_exposure)
                resize_exp = cv2.resize(norm_exp, (720, 480))
                accumulated_heatmap = cv2.applyColorMap(resize_exp, cv2.COLORMAP_JET)

                # accumulated_heatmap = cv2.applyColorMap(
                #     cv2.resize(vid_heatmap_normalize(final_exposure), (720, 480)),
                #     cv2.COLORMAP_JET)
                cv2.imshow('acc heatmap', accumulated_heatmap)


                # current heatmap (5 sec)
                if current_exposure is None or current_dash_time - prev_dash_time > 8:
                    current_exposure = np.zeros((im0.shape[0], im0.shape[1]), dtype=np.float)
                    prev_dash_time = current_dash_time
                current_exposure = current_exposure + new_exposures

                current_heatmap = cv2.applyColorMap(
                    cv2.resize(vid_heatmap_normalize(current_exposure), (720, 480)),
                    cv2.COLORMAP_JET)
                cv2.imshow('cur heatmap', current_heatmap)

                # Print time (inference + NMS)
                # print('%sDone. (%.3fs)' % (s, t2 - t1))

        # Stream results
        if view_img:
            cv2.imshow(p, cv2.resize(im0, (1280, 720)))
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'images':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                vid_writer.write(im0)

        # print('%sDone one frame. (%.3fs)' % (s, t2 - t1))

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    # theatmap = time.time()
    # print('Creating heatmap ... ')
    # alternative 1
    # create_heatmap(final_exposure)

    # alternative 2
    # plt.imshow(final_exposure, cmap='jet')
    # # # plt.show()
    # plt.savefig('heatmap plt.png', bbox_inches='tight')

    # alternative 3
    # color_image = cv2.applyColorMap(
    #     vid_heatmap_normalize(final_exposure),
    #     cv2.COLORMAP_JET)
    # # cv2.imshow('heatmap dash', color_image)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # cv2.imwrite('heatmap.jpg', color_image)

    # print('create heatmap time', (time.time() - theatmap))

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    # parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    # parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    # parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    # parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    # parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    # parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    # parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    # parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')

    parser.add_argument('--cfg', type=str, default='../Aerial Yolov3/cfg/yolov3-aerial.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='../Aerial PreTrained/aerial.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='../Aerial PreTrained/yolov3-aerial.weights', help='weights path')
    parser.add_argument('--source', type=str, default='../Aerial PreTrained/test video good/y2mate.com - A drones perspective of traffic jams_1080pFHR_Trim.mp4', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)

    with torch.no_grad():
        detect()
