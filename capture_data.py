from selenium import webdriver
import selenium.webdriver.common.keys as Keys
import time
import threading
from capture import capture_feed

# linking the webdriver
driver = webdriver.Chrome('/Users/vishrutsharma/Desktop/chromedriver')

# opening chrome in offline mode just to get the dino game
driver.get('https://www.google.com/')
time.sleep(2)
page = driver.find_element_by_class_name('offline')
page.send_keys(u'\ue00d')

# starting the feed capturing images
capture_feed.start()

while True:
    pass

