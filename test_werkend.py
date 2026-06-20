import depthai as dai
import numpy as np

CLASS_NAMES = ["class0", "class1", "class2", "class3"]  # pas later aan

blob_path = "/home/student/.cache/blobconverter/best_openvino_2022.1_6shave.blob"

pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

cam_out = cam.requestOutput(
    size=(640, 640),
    type=dai.ImgFrame.Type.RGB888p
)

nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(blob_path)

cam_out.link(nn.input)

nn_q = nn.out.createOutputQueue(maxSize=4, blocking=False)

pipeline.start()

print("Pipeline gestart ✔")

while True:
    msg = nn_q.get()

    output = msg.getFirstTensor()   # jouw v3 API geeft numpy.ndarray
    output = np.squeeze(output)     # (8, 8400)
    output = output.T               # (8400, 8)

    boxes = output[:, :4]           # x, y, w, h
    scores = output[:, 4:]          # class scores

    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    mask = confidences > 0.85

    print("Aantal ruwe detecties boven threshold:", np.sum(mask))

    for box, cls, conf in zip(boxes[mask][:10], class_ids[mask][:10], confidences[mask][:10]):
        x, y, w, h = box
        name = CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else str(cls)
        print(f"{name}: {conf:.2f} box=[{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]")