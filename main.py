from train import run


def main():

    conf_th = 0.05
    iou_th = 0.80
    data = 'sign_language.yaml'
    loss_function = 'ordinary'

    run(imgsz=640, epochs=100, data=data, weights='', cfg='models/yolov5s.yaml', patience=0, confidence_treshold=conf_th, iou_treshold=iou_th, loss_function=loss_function, class_wise_nms=True)

if __name__ == "__main__":
    main()