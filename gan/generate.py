import random


def generate_portrait():
    number = random.randint(1000, 9999)
    link = "/images/retrato_" + str(number) + ".png"
    return link
