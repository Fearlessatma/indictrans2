# Setup
•	Create env with python 3.10 version
py -3.10 -m venv myenv  
myenv\Scripts\Activate   

•	pip install torch transformers sacremoses pandas regex mock bitsandbytes scipy accelerate datasets sentencepiece mosestokenizer nltk cython 

•	pip install git+https://github.com/VarunGumma/IndicTransToolkit.git


•	python -c "import nltk; nltk.download('punkt')"

•	Clone IndicTrans2
git clone https://github.com/AI4Bharat/IndicTrans2.git
cd IndicTrans2/huggingface_interface

•	Clone IndicTransToolkit
git clone https://github.com/VarunGumma/IndicTransToolkit.git
cd IndicTransToolkit

•	Inside IndictransToolkit create main.py and paste the below code:
•	https://github.com/Fearlessatma/indictrans2/tree/main/huggingface_interface/IndicTransToolkit

# Solution for ModuleNotFoundError: No module named 'IndicTransToolkit.processor'

•	https://github.com/VarunGumma/IndicTransToolkit(reference)

•	pip install -e .

•	if the process fails 

1.	Download the installer:
Microsoft C++ Build Tools
2.	During installation, select: Desktop development with C++,Ensure that C++ build tools and Windows SDK are checked., Include the latest version of MSVC v14.x.
3.	Restart your system after the installation.
•	pip install -e .



# Languages
<table>
<tbody>
  <tr>
    <td>Assamese (asm_Beng)</td>
    <td>Kashmiri (Arabic) (kas_Arab)</td>
    <td>Punjabi (pan_Guru)</td>
  </tr>
  <tr>
    <td>Bengali (ben_Beng)</td>
    <td>Kashmiri (Devanagari) (kas_Deva)</td>
    <td>Sanskrit (san_Deva)</td>
  </tr>
  <tr>
    <td>Bodo (brx_Deva)</td>
    <td>Maithili (mai_Deva)</td>
    <td>Santali (sat_Olck)</td>
  </tr>
  <tr>
    <td>Dogri (doi_Deva)</td>
    <td>Malayalam (mal_Mlym)</td>
    <td>Sindhi (Arabic) (snd_Arab)</td>
  </tr>
  <tr>
    <td>English (eng_Latn)</td>
    <td>Marathi (mar_Deva)</td>
    <td>Sindhi (Devanagari) (snd_Deva)</td>
  </tr>
  <tr>
    <td>Konkani (gom_Deva)</td>
    <td>Manipuri (Bengali) (mni_Beng)</td>
    <td>Tamil (tam_Taml)</td>
  </tr>
  <tr>
    <td>Gujarati (guj_Gujr)</td>
    <td>Manipuri (Meitei) (mni_Mtei)</td>
    <td>Telugu (tel_Telu)</td>
  </tr>
  <tr>
    <td>Hindi (hin_Deva)</td>
    <td>Nepali (npi_Deva)</td>
    <td>Urdu (urd_Arab)</td>
  </tr>
  <tr>
    <td>Kannada (kan_Knda)</td>
    <td>Odia (ory_Orya)</td>
    <td></td>
  </tr>
</tbody>
</table>
