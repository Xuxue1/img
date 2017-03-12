
from captcha.image import ImageCaptcha
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import json
import base64
from io import BytesIO

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                 't', 'u', 'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img


class Recognition(object):

    def __init__(self, text_set=number+alphabet+ALPHABET, captcha_size=4, width=160, height=60):
        self.text_set = text_set
        self.captcha_size = captcha_size
        self.width = width
        self.height = height
        self.test_pos=0
        self.captcha_len = len(text_set)
        self.X = tf.placeholder(tf.float32, [None, self.width*self.height])
        self.Y = tf.placeholder(tf.float32, [None, self.captcha_size*self.captcha_len])
        self.keep_prob = tf.placeholder(tf.float32)
        self.x = tf.reshape(self.X, shape=[-1, self.height, self.width, 1])

        self.w_alpha = 0.01
        self.b_alpha = 0.1
        #定义三层卷积层

        self.w_c1 = tf.Variable(self.w_alpha * tf.random_normal([3, 3, 1, 32]))
        self.b_c1 = tf.Variable(self.b_alpha * tf.random_normal([32]))
        self.conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.x, self.w_c1, strides=[1, 1, 1, 1], padding='SAME'), self.b_c1))
        self.conv1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.conv1 = tf.nn.dropout(self.conv1, self.keep_prob)

        self.w_c2 = tf.Variable(self.w_alpha*tf.random_normal([3, 3, 32, 64]))
        self.b_c2 = tf.Variable(self.b_alpha*tf.random_normal([64]))
        self.conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv1, self.w_c2, strides=[1, 1, 1, 1], padding='SAME'), self.b_c2))
        self.conv2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.conv2 = tf.nn.dropout(self.conv2, self.keep_prob)

        self.w_c3 = tf.Variable(self.w_alpha*tf.random_normal([3, 3, 64, 64]))
        self.b_c3 = tf.Variable(self.b_alpha*tf.random_normal([64]))
        self.conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv2, self.w_c3, strides=[1, 1, 1, 1], padding='SAME'), self.b_c3))
        self.conv3 = tf.nn.max_pool(self.conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.conv3 = tf.nn.dropout(self.conv3, self.keep_prob)

        #全连接层

        self.w_d = tf.Variable(self.w_alpha*tf.random_normal([8*32*40, 1024]))
        self.b_d = tf.Variable(self.b_alpha*tf.random_normal([1024]))
        self.dense = tf.reshape(self.conv3, [-1, self.w_d.get_shape().as_list()[0]])
        self.dense = tf.nn.relu(tf.add(tf.matmul(self.dense, self.w_d), self.b_d))
        self.dense = tf.nn.dropout(self.dense, self.keep_prob)

        self.w_out = tf.Variable(self.w_alpha*tf.random_normal([1024, self.captcha_size*self.captcha_len]))
        self.b_out = tf.Variable(self.b_alpha*tf.random_normal([self.captcha_size*self.captcha_len]))
        self.out = tf.add(tf.matmul(self.dense, self.w_out), self.b_out)

    def random_captcha_text(self):
        captcha_text = []
        for i in range(self.captcha_size):
            c = random.choice(self.text_set)
            captcha_text.append(c)
        return captcha_text

    def gen_captcha_text_and_image(self):
        image = ImageCaptcha()
        captcha_text = self.random_captcha_text()
        captcha_text = ''.join(captcha_text)
        captcha = image.generate(captcha_text)
        captcha_image = Image.open(captcha)
        captcha_image = captcha_image.resize((self.width, self.height), Image.ANTIALIAS)
        captcha_image = np.array(captcha_image)
        return captcha_text, captcha_image

    def get_next_img(self):
        img_x, img_y = self.result.pop()
        imgdata=base64.b64decode(img_x)
        img_x = Image.open(BytesIO(imgdata))
        img_x = img_x.resize((self.width, self.height), Image.ANTIALIAS)
        img_x = np.array(img_x)
        return img_y, img_x

    def text2vec(self, text):
        text_len = len(text)
        if text_len != self.captcha_size:
            raise ValueError("验证码长度不匹配")
        vector = np.zeros(self.captcha_len * self.captcha_size)

        def char2pos(c):
            k = ord(c)-48
            if k > 9:
                k = ord(c)-55
                if k > 35:
                    k = ord(c) - 61
                    if k > 61:
                        raise ValueError('No Map')
            return k
        for i, c in enumerate(text):
            idx = i*self.captcha_len+char2pos(c)
            vector[idx] = 1
        return vector

    def vec2text(self, vec):
        char_pos = vec.nonzero()[0]
        text = []
        for i, c in enumerate(char_pos):
            char_idx = c % self.captcha_len
            if char_idx < 10:
                char_code = char_idx + ord('0')
            elif char_idx < 36:
                char_code = char_idx - 10 + ord('A')
            elif char_idx < 62:
                char_code = char_idx - 36 + ord('a')
            elif char_idx == 62:
                char_code = ord('_')
            else:
                raise ValueError('error')
            text.append(chr(char_code))
        return "".join(text)

    def get_next_batch(self, batch_size=128):
        batch_x = np.zeros([batch_size, self.width*self.height])
        batch_y = np.zeros([batch_size, self.captcha_len * self.captcha_size])
        for i in range(batch_size):
            text, image = self.get_next_img()
            image = convert2gray(image)
            batch_x[i, :] = image.flatten() / 255
            batch_y[i, :] = self.text2vec(text)
        return batch_x, batch_y

    def get_next_test_batch(self, batch_size=100):
        batch_x = np.zeros([batch_size, self.width*self.height])
        batch_y = np.zeros([batch_size, self.captcha_len * self.captcha_size])
        for i in range(batch_size):
            if self.test_pos >= len(self.test):
                self.test_pos = 0
            img_x, img_y = self.test[self.test_pos]
            img_data = base64.b64decode(img_x)
            img_x = Image.open(BytesIO(img_data))
            img_x = img_x.resize((self.width, self.height), Image.ANTIALIAS)
            img_x = np.array(img_x)
            image = convert2gray(img_x)
            batch_x[i, :] = image.flatten() / 255
            batch_y[i, :] = self.text2vec(img_y)
        return batch_x, batch_y

    def show_a_random_picture(self):
        text, image = self.gen_captcha_text_and_image()
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
        plt.imshow(image)
        plt.show()

    def read_img_from_log(self):
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
                    result.append((img_x, img_y['result']))
        self.result = result

    def read_test_data(self):
        file = open("test.log", encoding='utf-8')
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
                    result.append((img_x, img_y['result']))
        self.test = result

    def train(self):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.out))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        predict = tf.reshape(self.out, [-1, self.captcha_len, self.captcha_size])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, self.captcha_len, self.captcha_size]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            while True:
                batch_x, batch_y = self.get_next_batch(100)
                _, loss_ = sess.run([optimizer, loss], feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.75})
                print(step, loss_)
                if step % 10 == 0:
                    batch_x_test, batch_y_test = self.get_next_test_batch(100)
                    acc = sess.run(accuracy, feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1.0})
                    print('acc :'+str(acc))
                    if step != 0 and step % 500 == 0:
                        saver.save(sess, "model.model")
                        print("save")
                step += 1
            saver.save(sess, "model.model")


if __name__ == '__main__':
    r = Recognition()
    r.read_img_from_log()
    r.read_test_data()
    r.train()
