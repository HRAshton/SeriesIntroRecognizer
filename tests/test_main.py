from series_opening_recognizer.processors.audio_files import recognise_from_audio_files


def test_recognise_from_audio_files():
    files = [f'assets/audio_files/{i}.wav' for i in range(1, 8)]
    recognised_text = recognise_from_audio_files(files)

    assert recognised_text == []
