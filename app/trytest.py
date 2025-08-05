import json
import os
from pathlib import Path
from time import time
from datetime import datetime
import whisper

from app.core.bk_asr.asr_data import ASRData, ASRDataSeg

with open(f'/Users/yangshan/Downloads/srt/blender4.2.json') as file:
    raw_data = json.loads(file.read())
    segments = []
    segment_data = raw_data['segments']
    for raw_segment in segment_data:
        words = raw_segment['words']
        if words is not None and len(words) > 0:
            for word in words:
                segment = ASRDataSeg(
                    text=word["word"],
                    translated_text="",
                    start_time=word["start"]*1000,
                    end_time=word["end"]*1000,
                )
                segments.append(segment)
        else:
            segment = ASRDataSeg(
                text=raw_segment["text"],
                translated_text="",
                start_time=raw_segment["start"],
                end_time=raw_segment["end"],
            )
            segments.append(segment)
    asr = ASRData(segments=segments)
    print(asr.to_srt())


'''
videos = Path('/Users/yangshan/Downloads/blender_cu_test')
if not videos.exists() or not videos.is_dir():
    print(f"文件不存在{videos}")
    exit(1)
now = datetime.now()
print(f"Current datetime: {now}")
whisper_model = whisper.load_model('turbo', device='cpu')




for dirpath, dirnames, filenames in os.walk(videos):
    for filename in filenames:
        print(f'{dirpath}/{filename}')
        result = whisper_model.transcribe(f'{dirpath}/{filename}', word_timestamps=True, language='en',verbose=True)
        #print("result")
print(f"Time difference:{ (datetime.now()-now).total_seconds()}")
now = datetime.now()

for dirpath, dirnames, filenames in os.walk(videos):
    for filename in filenames:
        print(f'{dirpath}/{filename}')
        result = whisper_model.transcribe(f'{dirpath}/{filename}', language='en',verbose=True)
        #print(result)
print(f"Time difference:{ (datetime.now()-now).total_seconds()}")
'''
