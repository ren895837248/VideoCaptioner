import os
import shutil

listnamesfile = open('/Users/yangshan/Downloads/blenderculistnames')
names = listnamesfile.read()
lines = names.split('\n')
video_path = '/Users/yangshan/Downloads/blender_cu_new'
videolist = os.listdir(video_path)


count = 1
for filename in lines:
    try:
        shutil.move(f"{video_path}/{filename}.mp4", f"{video_path}/{count}.{filename}.mp4")
        count += 1
    except Exception as e:
        print(e)



# nolist = []
# for filename in lines:
#     num = 0
#     for videoname in videolist:
#         if videoname.startswith(filename):
#             num +=1
#             break
#     if num == 0:
#         nolist.append(filename)
#
# print(len(nolist))
#
# for name in nolist:
#     print(name)
# os.walk('/Users/yangshan/Downloads/blenderculistnames')
