
def sectostr(sec):
    sec = int(sec)
    minuts = sec // 60
    seconds = sec % 60
    time = ''
    if minuts > 0:
        time += str(minuts) + 'min'
    time += seconds + 's'
    return time
