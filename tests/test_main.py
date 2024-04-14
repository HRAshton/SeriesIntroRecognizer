from series_opening_recognizer.processors.audio_files import recognise_from_audio_files


def test_recognise_from_audio_files():
    files = [f'assets/audio_files/{i}.wav' for i in range(1, 8)]
    recognised = recognise_from_audio_files(files)

    for file, interval in recognised.items():
        print(file, interval, interval.end - interval.start)
        assert file in files
        assert interval.start >= 0
        assert interval.end >= 0
        assert interval.end - interval.start > 0
        assert 90 - (interval.end - interval.start) <= 1
