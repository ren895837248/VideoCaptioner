import os
import whisperx
import subprocess
from app.core.bk_asr.asr_data import ASRData, ASRDataSeg
from app.core.entities import SubtitleTask, SubtitleConfig
import datetime
from pathlib import Path

from app.common.config import cfg
import whisper

from app.core.subtitle_processor.split import SubtitleSplitter
from app.core.subtitle_processor.translate import TranslatorType, TranslatorFactory
import cons


def getArsData(raw_data) -> ASRData:
    """从JSON数据创建ASRData实例"""
    segments = []
    segment_data = raw_data['segments']
    for raw_segment in segment_data:
        words = raw_segment['words']
        if words is not None and len(words) > 0:
            for word in words:
                segment = ASRDataSeg(
                    text=word["word"],
                    translated_text="",
                    start_time=word["start"] * 1000,
                    end_time=word["end"] * 1000
                )
                segments.append(segment)
        else:
            segment = ASRDataSeg(
                text=raw_segment["text"],
                translated_text="",
                start_time=raw_segment["start"],
                end_time=raw_segment["end"]
            )
            segments.append(segment)
    return ASRData(segments)


def create_task(video_path: str, file_path: str) -> SubtitleTask:
    """创建字幕任务"""
    output_name = (
        Path(video_path)
        .stem.replace("【原始字幕】", "")
        .replace(f"【下载字幕】", "")
    )
    # 只在需要翻译时添加翻译服务后缀
    suffix = (
        f"-{cfg.translator_service.value.value}" if cfg.need_translate.value else ""
    )
    srt_dir = Path(f'{Path(video_path).parent}/subtitles')
    if not srt_dir.exists():
        srt_dir.mkdir()
    output_path = str(
        srt_dir / f"{output_name}.srt")
    split_type = "sentence"

    # 根据当前选择的LLM服务获取对应的配置
    llm_model = cfg.deepseek_model.value

    config = SubtitleConfig(
        # 翻译配置
        base_url=cons.base_url,
        api_key=cons.api_key,
        # llm_model=llm_model,#deepseek-chat
        llm_model="deepseek-chat",
        deeplx_endpoint=cfg.deeplx_endpoint.value,
        # 翻译服务
        translator_service=cfg.translator_service.value,
        # 字幕处理
        split_type='sentence',
        need_reflect=False,
        need_translate=True,
        need_optimize=False,
        thread_num=8,
        batch_size=cfg.batch_size.value,
        # 字幕布局、样式
        subtitle_layout=cfg.subtitle_layout.value,
        subtitle_style=None,
        # 字幕分割
        max_word_count_cjk=cfg.max_word_count_cjk.value,
        max_word_count_english=cfg.max_word_count_english.value,
        need_split=cfg.need_split.value,
        # 字幕翻译
        target_language=cfg.target_language.value.value,
        # 字幕优化
        need_remove_punctuation=cfg.needs_remove_punctuation.value,
        # 字幕提示
        custom_prompt_text=cfg.custom_prompt_text.value,
    )

    return SubtitleTask(
        queued_at=datetime.datetime.now(),
        subtitle_path=output_path,
        video_path=video_path,
        output_path=output_path,
        subtitle_config=config,
        need_next_task=False,
    )


videos = Path('/Users/yangshan/Downloads/blender_cu_new')
if not videos.exists() or not videos.is_dir():
    print(f"文件不存在{videos}")
    exit(1)

# whisper_model = whisper.load_model('turbo', device='cpu')
whisperx_model = whisperx.load_model("turbo", 'cpu', compute_type='float32')
model_a, metadata = whisperx.load_align_model(language_code='en', device='cpu')
for dirpath, dirnames, filenames in os.walk(videos):
    for filename in filenames:
        fullpath = f'{dirpath}/{filename}'
        if not Path(fullpath).suffix.endswith('mp4'):
            continue

        # result = whisper_model.transcribe(fullpath, word_timestamps=True, language='en', verbose=True)
        print(fullpath)
        audio = whisperx.load_audio(fullpath)
        result = whisperx_model.transcribe(audio, language='en', verbose=True)

        result = whisperx.align(result["segments"], model_a, metadata, audio, 'cpu', return_char_alignments=False)
        # with open('/Users/yangshan/Downloads/srt/blender4.2.json') as file:
        #     data = file.read()
        #     result = json.loads(data)
        asr_data = getArsData(result)

        task = create_task(fullpath, None)

        print(f"\n===========字幕处理任务开始===========")
        print(f"时间：{datetime.datetime.now()}")

        # 字幕文件路径检查、对断句字幕路径进行定义
        subtitle_path = task.subtitle_path

        subtitle_config = task.subtitle_config

        # asr_data = ASRData.from_subtitle_file(subtitle_path)

        base_url = "https://api.videocaptioner.cn/v1"
        api_key = "sk-MinI5WMRq3eL7Ew4vsvgYIowdXFpc4vbX0vFfHgjGCp8uMtN"
        os.environ['OPENAI_BASE_UR'] = base_url
        os.environ['OPENAI_API_KEY'] = api_key

        if subtitle_config.need_split and not asr_data.is_word_timestamp():
            asr_data.split_to_word_segments()
        # 2. 重新断句（对于字词级字幕）
        if asr_data.is_word_timestamp():
            print("正在字幕断句...")
            splitter = SubtitleSplitter(
                thread_num=subtitle_config.thread_num,
                model=subtitle_config.llm_model,
                temperature=0.3,
                timeout=60,
                retry_times=1,
                split_type=subtitle_config.split_type,
                max_word_count_cjk=subtitle_config.max_word_count_cjk,
                max_word_count_english=subtitle_config.max_word_count_english,
            )
            asr_data = splitter.split_subtitle(asr_data)
            # asr_data.save(save_path=split_path)

        # 4. 翻译字幕
        custom_prompt = subtitle_config.custom_prompt_text
        if subtitle_config.need_translate:
            print("正在翻译字幕...")
            finished_subtitle_length = 0  # 重置计数器
            # os.environ["DEEPLX_ENDPOINT"] = subtitle_config.deeplx_endpoint
            translator = TranslatorFactory.create_translator(
                translator_type=TranslatorType.OPENAI,
                thread_num=subtitle_config.thread_num,
                batch_num=subtitle_config.batch_size,
                target_language=subtitle_config.target_language,
                model=subtitle_config.llm_model,
                custom_prompt=custom_prompt,
                is_reflect=subtitle_config.need_reflect,
                update_callback=None,
            )

            asr_data = translator.translate_subtitle(asr_data)
            # 移除末尾标点符号
            if subtitle_config.need_remove_punctuation:
                asr_data.remove_punctuation()

            # 保存翻译结果(单语、双语)
            # if False:
            #     for subtitle_layout in ["原文在上", "译文在上", "仅原文", "仅译文"]:
            #         save_path = str(
            #             Path(task.subtitle_path).parent
            #             / f"{Path(task.video_path).stem}-{subtitle_layout}.srt"
            #         )
            #         asr_data.save(
            #             save_path=save_path,
            #             ass_style=subtitle_config.subtitle_style,
            #             layout=subtitle_layout,
            #         )
            #         print(f"字幕保存到 {save_path}")

        # 5. 保存字幕
        asr_data.save(
            save_path=task.output_path,
            ass_style=subtitle_config.subtitle_style,
            layout=subtitle_config.subtitle_layout,
        )
        print(f"字幕保存到 {task.output_path}")

        print(f'合并字幕到视频')
        # ffmpeg.input(f'{dirpath}/{filename}').filter('subtitles', task.output_path).output(
        #     f'{dirpath}/new_{filename}').run()

        # stream = ffmpeg.input(fullpath)
        # stream = stream.input(task.output_path).filter('subtitles')
        # ffmpeg.output(stream, f'{dirpath}/new_{filename}').run()
        if not Path(f'{dirpath}/withsubtitle').exists():
            Path(f'{dirpath}/withsubtitle').mkdir()
        process = subprocess.Popen([
            'ffmpeg',
            '-y',
            '-i', fullpath,
            '-i', task.output_path,
            '-vf', f"subtitles={task.output_path}",
            f'{dirpath}/withsubtitle/{Path(filename).stem}_withsubtitle.{Path(filename).suffix}'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f'字幕合成失败{stderr}')
        else:
            print(f'{fullpath}视频字幕处理完成')


        print(f'移动处理过的视频{fullpath}')
        if not Path(f'{dirpath}/hasprocessed').exists():
            Path(f'{dirpath}/hasprocessed').mkdir()

        process = subprocess.Popen([
            'mv',
            f'{fullpath}',
            f'{dirpath}/hasprocessed/{filename}'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f'移动处理过的视频失败：{stderr}')
        else:
            print(f'{fullpath}移动处理过的视频完成')
