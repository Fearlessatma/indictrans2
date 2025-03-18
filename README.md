•	Create env with python 3.10 version
py -3.10 -m venv myenv  
myenv\Scripts\Activate   

•	pip install torch transformers sacremoses pandas regex mock bitsandbytes scipy accelerate datasets sentencepiece mosestokenizer nltk

•	pip install git+https://github.com/VarunGumma/IndicTransToolkit.git


•	python -c "import nltk; nltk.download('punkt')"

•	Clone IndicTrans2
git clone https://github.com/AI4Bharat/IndicTrans2.git
cd IndicTrans2/huggingface_interface

•	Clone IndicTransToolkit
git clone https://github.com/VarunGumma/IndicTransToolkit.git
cd IndicTransToolkit

•	Inside IndictransToolkit create main.py and paste the below code:


# Solution for ModuleNotFoundError: No module named 'IndicTransToolkit.processor'

•	https://github.com/VarunGumma/IndicTransToolkit(reference)

•	pip install -e .

•	if the process fails 

1.	Download the installer:
Microsoft C++ Build Tools
2.	During installation, select: Desktop development with C++,Ensure that C++ build tools and Windows SDK are checked., Include the latest version of MSVC v14.x.
3.	Restart your system after the installation.
•	pip install -e .
