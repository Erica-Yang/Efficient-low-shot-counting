from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import cv2
CONFIG_PATH = "/root/loca/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"    #源码自带的配置文件
CHECKPOINT_PATH = "/root/loca/GroundingDINO/groundingdino_swint_ogc.pth"   #下载的权重文件
DEVICE = "cpu"   #可以选择cpu/cuda
IMAGE_PATH = "/root/loca/datasets/FSC147/images_384_VarV2/250.jpg"    #用户设置的需要读取image的路径
TEXT_PROMPT = "apple. flower. grape"    #用户给出的文本提示
BOX_TRESHOLD = 0.35     #源码给定的边界框判定阈值
TEXT_TRESHOLD = 0.25    #源码给定的文本端获取关键属性阈值
image_source, image = load_image(IMAGE_PATH)
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=DEVICE,
)
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("/root/loca/self_test/imgs/250.jpg", annotated_frame)
