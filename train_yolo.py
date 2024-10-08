#Código de referência https://github.com/ultralytics/ultralytics e https://github.com/inteligenciamilgrau/treinando_yolov8/blob/main/detectar_capturando_tela.py
#https://docs.ultralytics.com/pt/tasks/detect/#how-do-i-train-a-yolo11-model-on-my-custom-dataset
#https://docs.ultralytics.com/modes/train/#train-settings
#https://docs.ultralytics.com/modes/predict/#obb
from ultralytics import YOLO

# para marcar as imagens
# https://www.makesense.ai/

def main():
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="data.yaml", epochs=30, device=0, batch=-1)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # path = model.export(format="onnx")  # export the model to ONNX format
    # print("path", path)


if __name__ == '__main__':
    # freeze_support()
    main()