# Web Scrabbing and Lyrics Generating

In this self-developed project, my aim was to generate lyric by using lyrics of all discography of a given performer. I developed my model using Bob Dylan lyrics but it is open for new trials. 

At first step, I parsed lyrics from a web page via beautifulsoup package and then cleaned as well as prepared them for model development. 
After that I created a bidirectional LSTM model with a couple of layers, then trained it with a hundred iteration. Eventually, I provide initial words for prediction to the trained model and it provides prodicted a hundred more words addition to my initial words. 

**For trying prediction results with your initial words using pre-trained Bob-Dylan model;**
  - First it is required to either clone the repo or download (bob_dylan.h5, bobdylan_input_sequences.csv, bobdylan_tokenizer.joblib, PredictBobDylanLyrics.py) necessary files
  - Run **PredictBobDylanLyrics.py** file 
  - It will request you to write a few words to trigger the predictions based on your initial words

**For trying to train model with differen performer;**
  - Open **Lyrics_Generator.ipynb** file and run code cells in sequence,
  - Write the performer name when it will be requested at 5th code cell in required format, which is write your input in lowercase and put '-' between name(s) and surname(s) (ex: bob-dylan)
  - Let the model train with new lyrics, afterwards it will again request you to start generating lyrics with a few words, then it will generate a hundred more words to extend lyrics. 

***As a note, I parsed lyric data from a randomly found website https://sarki.alternatifim.com/ so it may have some rights for commercial use!***
