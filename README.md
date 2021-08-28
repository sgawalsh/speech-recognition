# speech-recognition

This project implements a training and decoding pipeline for speech recognition, trained using the LibriSpeech dataset.

Before the model can be trained, the data is processed from its initial ```.flac``` format, into the [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) respresentation of the waveform, which is associated with a transcription of the audio file to be used as target data during training.

The model is based on the model described in the [Wav2Letter](https://arxiv.org/pdf/1609.03193.pdf) paper by Facebook AI Research and is found in the ```models.py``` file. The application supports training the model on either a letter output, or phoneme output, according to the value selected in the ```train_batch``` function in ```wav2letter_torch.py```.

The project also implements several decoders in ```decoder.py```, in order to decode the output from our model trained using [CTC Loss](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html). A user can compare outputs between greedy, beam, and log beam decoder functions, as well as compare word error rates using the ```wer``` function. 

A Youtube series detailing the steps involved throughout the data pipeline can be found [here](https://www.youtube.com/playlist?list=PL3om9a5CvNUkpflccDZgr4EoUQpJLHdQ2).
