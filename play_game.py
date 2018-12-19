from selenium import webdriver
import selenium.webdriver.common.keys as Keys
import time
import threading
from capture import capture_feed
from network import ai
 
driver = webdriver.Chrome('/Users/vishrutsharma/Desktop/chromedriver')

# internet connection must be off
driver.get('http://www.google.com/')
time.sleep(2)

# main page to send key commands to
page = driver.find_element_by_class_name('offline')

# start game
page.send_keys(u'\ue00d')

# controls the dinosaur
while True:
    ai.predict(page) 