
﻿# Speech To Braille

## Table of contents
* [Description](#Description)
* [Technologies](#Technologies)
* [Setup](#Setup)
* [Execution](#Execution)

## Description
The aim is to successfully convert English speech to Braille script. To do this we will need to follow 3 key steps:
1. Accpting an audio file from the user to be converted. The audio file can be .wav or .mp3 format.
2. Converting the loaded audio to text. We do this using AssemblyAI's API for conversion. The result is then stored in a text file
3. Finally, we convert the text file to Braille script and stores it in a .png file so that it is printer friendly.


## Technologies
![Assembly AI](https://img.shields.io/badge/Assembly--AI-API-orange)
![Python](https://img.shields.io/badge/Python-3.10.7-blueviolet)

## Setup
This project has been written **Python Version 3.10.7**

To run the this project please run the following lines in the command prompt terminal and install the dependencies
```
pip install Pillow
pip install requests
pip install pydub
```
## Execution
- Step 1 : Download the Final App Folder in your Local System
- Step 2 : Open Command Prompt terminal in that directory/file
- Step 3 : Run  the following FinalCODE.py for recording , transcribing and conversion to Braille Script

