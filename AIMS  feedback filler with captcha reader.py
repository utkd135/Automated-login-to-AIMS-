import os
import numpy as np
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import selenium
from selenium.webdriver.common.action_chains import ActionChains
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException


Roll_Number = ''
Password = ''

# Path to store screenshot of 2nd Captcha
path_2ndcaptcha = r''  # path to ss of 2nd captcha

# Path to the ML model
path_to_model = r''  # path to model


def check_exists_by_xpath(xpath):
    try:
        webdriver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return True


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8192, 128),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(256, 51),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.network(x)
        return x


def bgr2rgb2gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def centerCrop(img):
    w, h = 220, 50
    a, b = img.shape
    center = a / 2, b / 2
    x = center[1] - w / 2  # width
    y = center[0] - h / 2  # height
    img = img[int(y):int(y + 50), int(x):int(x + 220)]
    return img


def coordinates(img):
    m, n = img.shape
    state = 0
    threshold = 200
    tmp1 = 0
    tmp2 = 0
    padd = 5
    list_break = []
    max_vals = img.max(axis=0, keepdims=True)
    for j in range(n):
        if max_vals[0, j] > threshold:
            if state == 0:
                tmp2 = j
                list_break.append(tmp1 + padd)
                list_break.append(tmp2 - padd)
            else:
                tmp1 = j

            state = 1
        else:
            state = 0
    list_break.pop(0)
    if tmp1 + padd > 220:
        list_break.append(220)
    else:
        list_break.append(int(tmp1 + padd))
    return list_break


def captcha2(path_2ndcaptcha, path_to_model):
    a = 0
    img = cv2.imread(path_2ndcaptcha)
    img = bgr2rgb2gray(img)
    img = centerCrop(img)
    lb = coordinates(img)

    if len(lb) != 10:
        print('will give error')

    labels_map = {
        0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8", 7: "9", 8: "A", 9: "B", 10: "C", 11: "D", 12: "E",
        13: "F",
        14: "G", 15: "H", 16: "J", 17: "K", 18: "L", 19: "M", 20: "P", 21: "Q", 22: "R", 23: "S", 24: "T", 25: "U",
        26: "V",
        27: "W", 28: "X", 29: "Y", 30: "a", 31: "b", 32: "c", 33: "d", 34: "e", 35: "f", 36: "h", 37: "j", 38: "k",
        39: "m",
        40: "n", 41: "p", 42: "q", 43: "r", 44: "s", 45: "t", 46: "u", 47: "v", 48: "w", 49: "x", 50: "y",
    }

    # Path to the ML model
    captcha_model = CNNModel()
    captcha_model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')), strict=True)
    captcha_model.eval()
    captcha = ""

    for i in range(5):
        a += 1
        img_crop = img[int(1):int(1 + 50), int(lb[2 * i]): int(lb[2 * i + 1])]
        img_crop = cv2.resize(img_crop, (35, 35))
        img_crop = 2. * (img_crop - np.min(img_crop)) / np.ptp(img_crop) - 1
        img_crop = torch.tensor(img_crop, dtype=torch.float32)
        img1 = img_crop.unsqueeze(0).unsqueeze(0)
        output = captcha_model(img1)
        _, pred = torch.max(output.data, 1)
        captcha += str(labels_map[pred.item()])

    return captcha


def fill_feedback():
    academic = WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.XPATH, "//span[@title='Academic']"))
    )
    actions.click(on_element=academic)

    view_courses = WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.XPATH, "//span[@title='View My Courses']"))
    )
    actions.move_to_element(to_element=view_courses).click(on_element=view_courses)
    actions.perform()

    feedback = WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.XPATH, "//span[@class='iconLeft']"))
    )
    feedback_buttons = driver.find_elements_by_class_name('fb_status_change_icon')
    n = len(feedback_buttons)
    if n != 0:
        for i in range(n):
            feedback_wait = WebDriverWait(driver, 60).until(
                EC.presence_of_element_located((By.XPATH, "//span[@class='iconLeft']"))
            )

            feedback_button = driver.find_element_by_class_name('fb_status_change_icon')
            feedback_link = feedback_button.get_attribute('href')
            driver.get(feedback_link)

            feedback2 = WebDriverWait(driver, 60).until(
                EC.presence_of_element_located((By.XPATH, "//span[@class='iconLeft']"))
            )
            feedback2_buttons = driver.find_elements_by_class_name('fb_status_change_icon')
            m = len(feedback2_buttons)
            for j in range(m):
                feedback_wait = WebDriverWait(driver, 60).until(
                    EC.presence_of_element_located((By.XPATH, "//span[@class='iconLeft']"))
                )

                feedback2_button = driver.find_element_by_class_name('fb_status_change_icon')
                feedback2_link = feedback2_button.get_attribute('href')
                driver.get(feedback2_link)

                feedback_value = "'4.00'"
                feedback3 = WebDriverWait(driver, 60).until(
                    EC.presence_of_element_located((By.XPATH, f"//input[@value={feedback_value}]"))
                )

                Remarks = ["Great Course!", "Very Informative", "Enjoyable", "Wonderful", "Exciting"]
                driver.find_element_by_id('fbRemarks').send_keys(random.choice(Remarks))

                console_feedback_command = f'$("[value={feedback_value}]").click()'
                driver.execute_script(console_feedback_command)

                submit_button_id = "'savefb'"
                console_submit_command = f'$("[id={submit_button_id}]").click()'
                driver.execute_script(console_feedback_command)

                actions.move_to_element(to_element=element).click(on_element=element)
                actions.perform()
    else:
        driver.close()


options = Options()
driver = webdriver.Chrome(executable_path=r'C:\Users\utkar\Downloads\chromedriver_win32\chromedriver.exe',
                          options=options)

# opening site
driver.get("https://aims.iith.ac.in/aims/")

# Detecting presence of Fields
uid = WebDriverWait(driver, 60).until(
    EC.presence_of_element_located((By.ID, "uid")))

# entering password and id
uid.send_keys(Roll_Number)
pwd = driver.find_element_by_id("pswrd")
pwd.send_keys(Password)

# captcha filling
captcha_img = driver.find_element_by_id("appCaptchaLoginImg")
tmp = captcha_img.get_attribute('src')
cap1 = tmp[-5:]
captcha1 = driver.find_element_by_id("captcha")
captcha1.send_keys(cap1)

# Clicking Submit button
button1 = driver.find_element_by_id("login")
button1.click()
actions = ActionChains(driver)

for test_loop in range(10):
    time.sleep(4 * (test_loop + 1))

    # Detecting presence of 2nd captcha
    captcha_name = WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="appCaptchaLoginImg"]')))
    captcha_name = driver.find_element_by_xpath('//*[@id="appCaptchaLoginImg"]')
    captcha_name.screenshot(path_2ndcaptcha)

    # Captcha
    captcha = captcha2(path_2ndcaptcha, path_to_model)

    # Entering the captcha
    captcha_place = driver.find_element_by_xpath('//*[@id="captcha"]')
    captcha_place.send_keys(captcha)

    # Clicking the Submit button
    submit = driver.find_element_by_xpath('//*[@id="submit"]')
    submit.click()

    time.sleep(4)

    if check_exists_by_xpath("//span[@title='Academic']"):
        break

    else:
        refresh_button = driver.find_element_by_id("loginCapchaRefresh")
        refresh_button.click()

fill_feedback()
