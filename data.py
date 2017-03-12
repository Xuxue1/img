import re
import json
from PIL import Image
import base64
from io import BytesIO

def filter_img():
    file = open("collected.log.20170309", encoding='utf-8')
    content = file.read()
    content = content.replace("\n","")
    imgs = re.findall(r'content.*?\}', content)
    result = []
    for img in imgs:
        img_x = re.findall(r'content:.*?result:', img)
        img_y = re.findall(r'result:.*?\}', img)
        img_x = img_x[0].replace("content:", "").replace("result:", "")
        img_y = json.loads(img_y[0].replace("result:", ""))
        if 'result' in img_y:
            if len(img_y['result']) == 4:
                print(img_y['result'])
                img_x = base64.b64decode(img_x)
                img_x = Image.open(BytesIO(img_x))
                print(type(img_x))
                result.append(img_y['result'])
    print(len(result))


if __name__ == "__main__":
    p = []
    print(p.pop())