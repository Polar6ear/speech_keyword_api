from gtts import gTTS

text = "Yesterday I ate pizza and biryani. Later I had ice cream and coffee."

tts = gTTS(text=text, lang='en')
tts.save('test_audio1.mp3')
