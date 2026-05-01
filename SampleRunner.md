```python
!git clone https://github.com/BaristaBandits/Distribution_Estimation.git
```

    Cloning into 'Distribution_Estimation'...
    remote: Enumerating objects: 229, done.[K
    remote: Counting objects: 100% (73/73), done.[K
    remote: Compressing objects: 100% (73/73), done.[K
    remote: Total 229 (delta 40), reused 0 (delta 0), pack-reused 156 (from 1)[K
    Receiving objects: 100% (229/229), 104.05 KiB | 1.46 MiB/s, done.
    Resolving deltas: 100% (127/127), done.



```python
!pip install -r Distribution_Estimation/requirements.txt
```

    Collecting numpy==1.26.4 (from -r Distribution_Estimation/requirements.txt (line 1))
      Downloading numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m61.0/61.0 kB[0m [31m697.4 kB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: matplotlib==3.10.0 in /usr/local/lib/python3.12/dist-packages (from -r Distribution_Estimation/requirements.txt (line 2)) (3.10.0)
    Requirement already satisfied: tqdm==4.67.3 in /usr/local/lib/python3.12/dist-packages (from -r Distribution_Estimation/requirements.txt (line 3)) (4.67.3)
    Requirement already satisfied: scikit-learn==1.6.1 in /usr/local/lib/python3.12/dist-packages (from -r Distribution_Estimation/requirements.txt (line 4)) (1.6.1)
    Collecting gensim==4.3.3 (from -r Distribution_Estimation/requirements.txt (line 5))
      Downloading gensim-4.3.3-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (8.1 kB)
    Requirement already satisfied: datasets==4.0.0 in /usr/local/lib/python3.12/dist-packages (from -r Distribution_Estimation/requirements.txt (line 6)) (4.0.0)
    Requirement already satisfied: nltk==3.9.1 in /usr/local/lib/python3.12/dist-packages (from -r Distribution_Estimation/requirements.txt (line 7)) (3.9.1)
    Requirement already satisfied: transformers==5.0.0 in /usr/local/lib/python3.12/dist-packages (from -r Distribution_Estimation/requirements.txt (line 8)) (5.0.0)
    Requirement already satisfied: torch==2.10.0 in /usr/local/lib/python3.12/dist-packages (from -r Distribution_Estimation/requirements.txt (line 9)) (2.10.0+cpu)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib==3.10.0->-r Distribution_Estimation/requirements.txt (line 2)) (1.3.3)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.12/dist-packages (from matplotlib==3.10.0->-r Distribution_Estimation/requirements.txt (line 2)) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib==3.10.0->-r Distribution_Estimation/requirements.txt (line 2)) (4.62.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib==3.10.0->-r Distribution_Estimation/requirements.txt (line 2)) (1.5.0)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib==3.10.0->-r Distribution_Estimation/requirements.txt (line 2)) (26.1)
    Requirement already satisfied: pillow>=8 in /usr/local/lib/python3.12/dist-packages (from matplotlib==3.10.0->-r Distribution_Estimation/requirements.txt (line 2)) (11.3.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib==3.10.0->-r Distribution_Estimation/requirements.txt (line 2)) (3.3.2)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.12/dist-packages (from matplotlib==3.10.0->-r Distribution_Estimation/requirements.txt (line 2)) (2.9.0.post0)
    Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn==1.6.1->-r Distribution_Estimation/requirements.txt (line 4)) (1.16.3)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn==1.6.1->-r Distribution_Estimation/requirements.txt (line 4)) (1.5.3)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn==1.6.1->-r Distribution_Estimation/requirements.txt (line 4)) (3.6.0)
    Collecting scipy>=1.6.0 (from scikit-learn==1.6.1->-r Distribution_Estimation/requirements.txt (line 4))
      Downloading scipy-1.13.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (60 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m60.6/60.6 kB[0m [31m1.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: smart-open>=1.8.1 in /usr/local/lib/python3.12/dist-packages (from gensim==4.3.3->-r Distribution_Estimation/requirements.txt (line 5)) (7.6.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (3.29.0)
    Requirement already satisfied: pyarrow>=15.0.0 in /usr/local/lib/python3.12/dist-packages (from datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (18.1.0)
    Requirement already satisfied: dill<0.3.9,>=0.3.0 in /usr/local/lib/python3.12/dist-packages (from datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (0.3.8)
    Requirement already satisfied: pandas in /usr/local/lib/python3.12/dist-packages (from datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (2.2.2)
    Requirement already satisfied: requests>=2.32.2 in /usr/local/lib/python3.12/dist-packages (from datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (2.32.4)
    Requirement already satisfied: xxhash in /usr/local/lib/python3.12/dist-packages (from datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (3.6.0)
    Requirement already satisfied: multiprocess<0.70.17 in /usr/local/lib/python3.12/dist-packages (from datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (0.70.16)
    Requirement already satisfied: fsspec<=2025.3.0,>=2023.1.0 in /usr/local/lib/python3.12/dist-packages (from fsspec[http]<=2025.3.0,>=2023.1.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (2025.3.0)
    Requirement already satisfied: huggingface-hub>=0.24.0 in /usr/local/lib/python3.12/dist-packages (from datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (1.11.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.12/dist-packages (from datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (6.0.3)
    Requirement already satisfied: click in /usr/local/lib/python3.12/dist-packages (from nltk==3.9.1->-r Distribution_Estimation/requirements.txt (line 7)) (8.3.3)
    Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.12/dist-packages (from nltk==3.9.1->-r Distribution_Estimation/requirements.txt (line 7)) (2025.11.3)
    Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /usr/local/lib/python3.12/dist-packages (from transformers==5.0.0->-r Distribution_Estimation/requirements.txt (line 8)) (0.22.2)
    Requirement already satisfied: typer-slim in /usr/local/lib/python3.12/dist-packages (from transformers==5.0.0->-r Distribution_Estimation/requirements.txt (line 8)) (0.24.0)
    Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.12/dist-packages (from transformers==5.0.0->-r Distribution_Estimation/requirements.txt (line 8)) (0.7.0)
    Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.12/dist-packages (from torch==2.10.0->-r Distribution_Estimation/requirements.txt (line 9)) (4.15.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch==2.10.0->-r Distribution_Estimation/requirements.txt (line 9)) (75.2.0)
    Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch==2.10.0->-r Distribution_Estimation/requirements.txt (line 9)) (1.14.0)
    Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch==2.10.0->-r Distribution_Estimation/requirements.txt (line 9)) (3.6.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch==2.10.0->-r Distribution_Estimation/requirements.txt (line 9)) (3.1.6)
    Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /usr/local/lib/python3.12/dist-packages (from fsspec[http]<=2025.3.0,>=2023.1.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (3.13.5)
    Requirement already satisfied: hf-xet<2.0.0,>=1.4.3 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.24.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (1.4.3)
    Requirement already satisfied: httpx<1,>=0.23.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.24.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (0.28.1)
    Requirement already satisfied: typer in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.24.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (0.24.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.7->matplotlib==3.10.0->-r Distribution_Estimation/requirements.txt (line 2)) (1.17.0)
    Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests>=2.32.2->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (3.4.7)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests>=2.32.2->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (3.13)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests>=2.32.2->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (2.5.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests>=2.32.2->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (2026.4.22)
    Requirement already satisfied: wrapt in /usr/local/lib/python3.12/dist-packages (from smart-open>=1.8.1->gensim==4.3.3->-r Distribution_Estimation/requirements.txt (line 5)) (2.1.2)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch==2.10.0->-r Distribution_Estimation/requirements.txt (line 9)) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch==2.10.0->-r Distribution_Estimation/requirements.txt (line 9)) (3.0.3)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (2026.1)
    Requirement already satisfied: aiohappyeyeballs>=2.5.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (2.6.1)
    Requirement already satisfied: aiosignal>=1.4.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (1.4.0)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (26.1.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (1.8.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (6.7.1)
    Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (0.4.1)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (1.23.0)
    Requirement already satisfied: anyio in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface-hub>=0.24.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (4.13.0)
    Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface-hub>=0.24.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (1.0.9)
    Requirement already satisfied: h11>=0.16 in /usr/local/lib/python3.12/dist-packages (from httpcore==1.*->httpx<1,>=0.23.0->huggingface-hub>=0.24.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (0.16.0)
    Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub>=0.24.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (1.5.4)
    Requirement already satisfied: rich>=12.3.0 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub>=0.24.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (13.9.4)
    Requirement already satisfied: annotated-doc>=0.0.2 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub>=0.24.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (0.0.4)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.12/dist-packages (from rich>=12.3.0->typer->huggingface-hub>=0.24.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (4.0.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.12/dist-packages (from rich>=12.3.0->typer->huggingface-hub>=0.24.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (2.20.0)
    Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.12/dist-packages (from markdown-it-py>=2.2.0->rich>=12.3.0->typer->huggingface-hub>=0.24.0->datasets==4.0.0->-r Distribution_Estimation/requirements.txt (line 6)) (0.1.2)
    Downloading numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.0 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m18.0/18.0 MB[0m [31m44.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading gensim-4.3.3-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (26.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m26.6/26.6 MB[0m [31m36.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading scipy-1.13.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (38.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m38.2/38.2 MB[0m [31m14.9 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: numpy, scipy, gensim
      Attempting uninstall: numpy
        Found existing installation: numpy 2.0.2
        Uninstalling numpy-2.0.2:
          Successfully uninstalled numpy-2.0.2
      Attempting uninstall: scipy
        Found existing installation: scipy 1.16.3
        Uninstalling scipy-1.16.3:
          Successfully uninstalled scipy-1.16.3
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    tobler 0.14.0 requires numpy>=2.0, but you have numpy 1.26.4 which is incompatible.
    pytensor 2.38.2 requires numpy>=2.0, but you have numpy 1.26.4 which is incompatible.
    shap 0.51.0 requires numpy>=2, but you have numpy 1.26.4 which is incompatible.
    jax 0.7.2 requires numpy>=2.0, but you have numpy 1.26.4 which is incompatible.
    xarray-einstats 0.10.0 requires numpy>=2.0, but you have numpy 1.26.4 which is incompatible.
    jaxlib 0.7.2 requires numpy>=2.0, but you have numpy 1.26.4 which is incompatible.
    opencv-python-headless 4.13.0.92 requires numpy>=2; python_version >= "3.9", but you have numpy 1.26.4 which is incompatible.
    opencv-contrib-python 4.13.0.92 requires numpy>=2; python_version >= "3.9", but you have numpy 1.26.4 which is incompatible.
    tifffile 2026.4.11 requires numpy>=2.0, but you have numpy 1.26.4 which is incompatible.
    access 1.1.10.post3 requires scipy>=1.14.1, but you have scipy 1.13.1 which is incompatible.
    opencv-python 4.13.0.92 requires numpy>=2; python_version >= "3.9", but you have numpy 1.26.4 which is incompatible.
    tsfresh 0.21.1 requires scipy>=1.14.0; python_version >= "3.10", but you have scipy 1.13.1 which is incompatible.
    rasterio 1.5.0 requires numpy>=2, but you have numpy 1.26.4 which is incompatible.[0m[31m
    [0mSuccessfully installed gensim-4.3.3 numpy-1.26.4 scipy-1.13.1





```bash
%%bash
sudo apt-get update
sudo apt-get install -y texlive-xetex texlive-latex-extra texlive-fonts-recommended
sudo apt-get install -y fonts-freefont-ttf
```

    Get:1 https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/ InRelease [3,632 B]
    Get:2 https://cli.github.com/packages stable InRelease [3,917 B]
    Get:3 https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/ Packages [89.0 kB]
    Get:4 http://security.ubuntu.com/ubuntu jammy-security InRelease [129 kB]
    Get:5 https://cli.github.com/packages stable/main amd64 Packages [356 B]
    Hit:6 http://archive.ubuntu.com/ubuntu jammy InRelease
    Get:7 https://r2u.stat.illinois.edu/ubuntu jammy InRelease [6,555 B]
    Get:8 http://archive.ubuntu.com/ubuntu jammy-updates InRelease [128 kB]
    Get:9 https://r2u.stat.illinois.edu/ubuntu jammy/main amd64 Packages [2,990 kB]
    Get:10 http://security.ubuntu.com/ubuntu jammy-security/main amd64 Packages [3,864 kB]
    Get:11 http://archive.ubuntu.com/ubuntu jammy-backports InRelease [127 kB]
    Get:12 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 Packages [4,237 kB]
    Get:13 https://r2u.stat.illinois.edu/ubuntu jammy/main all Packages [10.1 MB]
    Get:14 http://security.ubuntu.com/ubuntu jammy-security/universe amd64 Packages [1,292 kB]
    Get:15 http://security.ubuntu.com/ubuntu jammy-security/restricted amd64 Packages [6,862 kB]
    Get:16 http://security.ubuntu.com/ubuntu jammy-security/multiverse amd64 Packages [61.6 kB]
    Get:17 http://archive.ubuntu.com/ubuntu jammy-updates/restricted amd64 Packages [7,225 kB]
    Get:18 http://archive.ubuntu.com/ubuntu jammy-updates/multiverse amd64 Packages [86.0 kB]
    Get:19 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 Packages [1,601 kB]
    Get:20 http://archive.ubuntu.com/ubuntu jammy-backports/main amd64 Packages [82.7 kB]
    Get:21 http://archive.ubuntu.com/ubuntu jammy-backports/universe amd64 Packages [35.7 kB]
    Ign:22 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy InRelease
    Ign:23 https://ppa.launchpadcontent.net/ubuntugis/ppa/ubuntu jammy InRelease
    Ign:22 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy InRelease
    Ign:23 https://ppa.launchpadcontent.net/ubuntugis/ppa/ubuntu jammy InRelease
    Ign:22 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy InRelease
    Ign:23 https://ppa.launchpadcontent.net/ubuntugis/ppa/ubuntu jammy InRelease
    Err:22 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy InRelease
      Could not connect to ppa.launchpadcontent.net:443 (185.125.190.80), connection timed out
    Err:23 https://ppa.launchpadcontent.net/ubuntugis/ppa/ubuntu jammy InRelease
      Unable to connect to ppa.launchpadcontent.net:443:
    Fetched 39.0 MB in 38s (1,037 kB/s)
    Reading package lists...
    Reading package lists...
    Building dependency tree...
    Reading state information...
    The following additional packages will be installed:
      dvisvgm fonts-droid-fallback fonts-lato fonts-lmodern fonts-noto-mono
      fonts-texgyre fonts-urw-base35 libapache-pom-java libcommons-logging-java
      libcommons-parent-java libfontbox-java libgs9 libgs9-common libidn12
      libijs-0.35 libjbig2dec0 libkpathsea6 libpdfbox-java libptexenc1 libruby3.0
      libsynctex2 libteckit0 libtexlua53 libtexluajit2 libwoff1 libzzip-0-13
      lmodern poppler-data preview-latex-style rake ruby ruby-net-telnet
      ruby-rubygems ruby-webrick ruby-xmlrpc ruby3.0 rubygems-integration t1utils
      teckit tex-common tex-gyre texlive-base texlive-binaries texlive-latex-base
      texlive-latex-recommended texlive-pictures texlive-plain-generic tipa
      xfonts-encodings xfonts-utils
    Suggested packages:
      fonts-noto fonts-freefont-otf | fonts-freefont-ttf libavalon-framework-java
      libcommons-logging-java-doc libexcalibur-logkit-java liblog4j1.2-java
      poppler-utils ghostscript fonts-japanese-mincho | fonts-ipafont-mincho
      fonts-japanese-gothic | fonts-ipafont-gothic fonts-arphic-ukai
      fonts-arphic-uming fonts-nanum ri ruby-dev bundler debhelper gv
      | postscript-viewer perl-tk xpdf | pdf-viewer xzdec
      texlive-fonts-recommended-doc texlive-latex-base-doc python3-pygments
      icc-profiles libfile-which-perl libspreadsheet-parseexcel-perl
      texlive-latex-extra-doc texlive-latex-recommended-doc texlive-luatex
      texlive-pstricks dot2tex prerex texlive-pictures-doc vprerex
      default-jre-headless tipa-doc
    The following NEW packages will be installed:
      dvisvgm fonts-droid-fallback fonts-lato fonts-lmodern fonts-noto-mono
      fonts-texgyre fonts-urw-base35 libapache-pom-java libcommons-logging-java
      libcommons-parent-java libfontbox-java libgs9 libgs9-common libidn12
      libijs-0.35 libjbig2dec0 libkpathsea6 libpdfbox-java libptexenc1 libruby3.0
      libsynctex2 libteckit0 libtexlua53 libtexluajit2 libwoff1 libzzip-0-13
      lmodern poppler-data preview-latex-style rake ruby ruby-net-telnet
      ruby-rubygems ruby-webrick ruby-xmlrpc ruby3.0 rubygems-integration t1utils
      teckit tex-common tex-gyre texlive-base texlive-binaries
      texlive-fonts-recommended texlive-latex-base texlive-latex-extra
      texlive-latex-recommended texlive-pictures texlive-plain-generic
      texlive-xetex tipa xfonts-encodings xfonts-utils
    0 upgraded, 53 newly installed, 0 to remove and 60 not upgraded.
    Need to get 182 MB of archives.
    After this operation, 571 MB of additional disk space will be used.
    Get:1 http://archive.ubuntu.com/ubuntu jammy/main amd64 fonts-droid-fallback all 1:6.0.1r16-1.1build1 [1,805 kB]
    Get:2 http://archive.ubuntu.com/ubuntu jammy/main amd64 fonts-lato all 2.0-2.1 [2,696 kB]
    Get:3 http://archive.ubuntu.com/ubuntu jammy/main amd64 poppler-data all 0.4.11-1 [2,171 kB]
    Get:4 http://archive.ubuntu.com/ubuntu jammy/universe amd64 tex-common all 6.17 [33.7 kB]
    Get:5 http://archive.ubuntu.com/ubuntu jammy/main amd64 fonts-urw-base35 all 20200910-1 [6,367 kB]
    Get:6 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libgs9-common all 9.55.0~dfsg1-0ubuntu5.13 [753 kB]
    Get:7 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libidn12 amd64 1.38-4ubuntu1 [60.0 kB]
    Get:8 http://archive.ubuntu.com/ubuntu jammy/main amd64 libijs-0.35 amd64 0.35-15build2 [16.5 kB]
    Get:9 http://archive.ubuntu.com/ubuntu jammy/main amd64 libjbig2dec0 amd64 0.19-3build2 [64.7 kB]
    Get:10 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libgs9 amd64 9.55.0~dfsg1-0ubuntu5.13 [5,032 kB]
    Get:11 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libkpathsea6 amd64 2021.20210626.59705-1ubuntu0.3 [60.6 kB]
    Get:12 http://archive.ubuntu.com/ubuntu jammy/main amd64 libwoff1 amd64 1.0.2-1build4 [45.2 kB]
    Get:13 http://archive.ubuntu.com/ubuntu jammy/universe amd64 dvisvgm amd64 2.13.1-1 [1,221 kB]
    Get:14 http://archive.ubuntu.com/ubuntu jammy/universe amd64 fonts-lmodern all 2.004.5-6.1 [4,532 kB]
    Get:15 http://archive.ubuntu.com/ubuntu jammy/main amd64 fonts-noto-mono all 20201225-1build1 [397 kB]
    Get:16 http://archive.ubuntu.com/ubuntu jammy/universe amd64 fonts-texgyre all 20180621-3.1 [10.2 MB]
    Get:17 http://archive.ubuntu.com/ubuntu jammy/universe amd64 libapache-pom-java all 18-1 [4,720 B]
    Get:18 http://archive.ubuntu.com/ubuntu jammy/universe amd64 libcommons-parent-java all 43-1 [10.8 kB]
    Get:19 http://archive.ubuntu.com/ubuntu jammy/universe amd64 libcommons-logging-java all 1.2-2 [60.3 kB]
    Get:20 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libptexenc1 amd64 2021.20210626.59705-1ubuntu0.3 [39.1 kB]
    Get:21 http://archive.ubuntu.com/ubuntu jammy/main amd64 rubygems-integration all 1.18 [5,336 B]
    Get:22 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 ruby3.0 amd64 3.0.2-7ubuntu2.12 [50.1 kB]
    Get:23 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 ruby-rubygems all 3.3.5-2ubuntu1.2 [228 kB]
    Get:24 http://archive.ubuntu.com/ubuntu jammy/main amd64 ruby amd64 1:3.0~exp1 [5,100 B]
    Get:25 http://archive.ubuntu.com/ubuntu jammy/main amd64 rake all 13.0.6-2 [61.7 kB]
    Get:26 http://archive.ubuntu.com/ubuntu jammy/main amd64 ruby-net-telnet all 0.1.1-2 [12.6 kB]
    Get:27 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 ruby-webrick all 1.7.0-3ubuntu0.2 [52.5 kB]
    Get:28 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 ruby-xmlrpc all 0.3.2-1ubuntu0.1 [24.9 kB]
    Get:29 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libruby3.0 amd64 3.0.2-7ubuntu2.12 [5,113 kB]
    Get:30 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libsynctex2 amd64 2021.20210626.59705-1ubuntu0.3 [55.8 kB]
    Get:31 http://archive.ubuntu.com/ubuntu jammy/universe amd64 libteckit0 amd64 2.5.11+ds1-1 [421 kB]
    Get:32 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libtexlua53 amd64 2021.20210626.59705-1ubuntu0.3 [120 kB]
    Get:33 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libtexluajit2 amd64 2021.20210626.59705-1ubuntu0.3 [267 kB]
    Get:34 http://archive.ubuntu.com/ubuntu jammy/universe amd64 libzzip-0-13 amd64 0.13.72+dfsg.1-1.1 [27.0 kB]
    Get:35 http://archive.ubuntu.com/ubuntu jammy/main amd64 xfonts-encodings all 1:1.0.5-0ubuntu2 [578 kB]
    Get:36 http://archive.ubuntu.com/ubuntu jammy/main amd64 xfonts-utils amd64 1:7.7+6build2 [94.6 kB]
    Get:37 http://archive.ubuntu.com/ubuntu jammy/universe amd64 lmodern all 2.004.5-6.1 [9,471 kB]
    Get:38 http://archive.ubuntu.com/ubuntu jammy/universe amd64 preview-latex-style all 12.2-1ubuntu1 [185 kB]
    Get:39 http://archive.ubuntu.com/ubuntu jammy/main amd64 t1utils amd64 1.41-4build2 [61.3 kB]
    Get:40 http://archive.ubuntu.com/ubuntu jammy/universe amd64 teckit amd64 2.5.11+ds1-1 [699 kB]
    Get:41 http://archive.ubuntu.com/ubuntu jammy/universe amd64 tex-gyre all 20180621-3.1 [6,209 kB]
    Get:42 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 texlive-binaries amd64 2021.20210626.59705-1ubuntu0.3 [9,861 kB]
    Get:43 http://archive.ubuntu.com/ubuntu jammy/universe amd64 texlive-base all 2021.20220204-1 [21.0 MB]
    Get:44 http://archive.ubuntu.com/ubuntu jammy/universe amd64 texlive-fonts-recommended all 2021.20220204-1 [4,972 kB]
    Get:45 http://archive.ubuntu.com/ubuntu jammy/universe amd64 texlive-latex-base all 2021.20220204-1 [1,128 kB]
    Get:46 http://archive.ubuntu.com/ubuntu jammy/universe amd64 libfontbox-java all 1:1.8.16-2 [207 kB]
    Get:47 http://archive.ubuntu.com/ubuntu jammy/universe amd64 libpdfbox-java all 1:1.8.16-2 [5,199 kB]
    Get:48 http://archive.ubuntu.com/ubuntu jammy/universe amd64 texlive-latex-recommended all 2021.20220204-1 [14.4 MB]
    Get:49 http://archive.ubuntu.com/ubuntu jammy/universe amd64 texlive-pictures all 2021.20220204-1 [8,720 kB]
    Get:50 http://archive.ubuntu.com/ubuntu jammy/universe amd64 texlive-latex-extra all 2021.20220204-1 [13.9 MB]
    Get:51 http://archive.ubuntu.com/ubuntu jammy/universe amd64 texlive-plain-generic all 2021.20220204-1 [27.5 MB]
    Get:52 http://archive.ubuntu.com/ubuntu jammy/universe amd64 tipa all 2:1.3-21 [2,967 kB]
    Get:53 http://archive.ubuntu.com/ubuntu jammy/universe amd64 texlive-xetex all 2021.20220204-1 [12.4 MB]
    Fetched 182 MB in 11s (15.9 MB/s)
    Selecting previously unselected package fonts-droid-fallback.
    (Reading database ... (Reading database ... 5%(Reading database ... 10%(Reading database ... 15%(Reading database ... 20%(Reading database ... 25%(Reading database ... 30%(Reading database ... 35%(Reading database ... 40%(Reading database ... 45%(Reading database ... 50%(Reading database ... 55%(Reading database ... 60%(Reading database ... 65%(Reading database ... 70%(Reading database ... 75%(Reading database ... 80%(Reading database ... 85%(Reading database ... 90%(Reading database ... 95%(Reading database ... 100%(Reading database ... 118194 files and directories currently installed.)
    Preparing to unpack .../00-fonts-droid-fallback_1%3a6.0.1r16-1.1build1_all.deb ...
    Unpacking fonts-droid-fallback (1:6.0.1r16-1.1build1) ...
    Selecting previously unselected package fonts-lato.
    Preparing to unpack .../01-fonts-lato_2.0-2.1_all.deb ...
    Unpacking fonts-lato (2.0-2.1) ...
    Selecting previously unselected package poppler-data.
    Preparing to unpack .../02-poppler-data_0.4.11-1_all.deb ...
    Unpacking poppler-data (0.4.11-1) ...
    Selecting previously unselected package tex-common.
    Preparing to unpack .../03-tex-common_6.17_all.deb ...
    Unpacking tex-common (6.17) ...
    Selecting previously unselected package fonts-urw-base35.
    Preparing to unpack .../04-fonts-urw-base35_20200910-1_all.deb ...
    Unpacking fonts-urw-base35 (20200910-1) ...
    Selecting previously unselected package libgs9-common.
    Preparing to unpack .../05-libgs9-common_9.55.0~dfsg1-0ubuntu5.13_all.deb ...
    Unpacking libgs9-common (9.55.0~dfsg1-0ubuntu5.13) ...
    Selecting previously unselected package libidn12:amd64.
    Preparing to unpack .../06-libidn12_1.38-4ubuntu1_amd64.deb ...
    Unpacking libidn12:amd64 (1.38-4ubuntu1) ...
    Selecting previously unselected package libijs-0.35:amd64.
    Preparing to unpack .../07-libijs-0.35_0.35-15build2_amd64.deb ...
    Unpacking libijs-0.35:amd64 (0.35-15build2) ...
    Selecting previously unselected package libjbig2dec0:amd64.
    Preparing to unpack .../08-libjbig2dec0_0.19-3build2_amd64.deb ...
    Unpacking libjbig2dec0:amd64 (0.19-3build2) ...
    Selecting previously unselected package libgs9:amd64.
    Preparing to unpack .../09-libgs9_9.55.0~dfsg1-0ubuntu5.13_amd64.deb ...
    Unpacking libgs9:amd64 (9.55.0~dfsg1-0ubuntu5.13) ...
    Selecting previously unselected package libkpathsea6:amd64.
    Preparing to unpack .../10-libkpathsea6_2021.20210626.59705-1ubuntu0.3_amd64.deb ...
    Unpacking libkpathsea6:amd64 (2021.20210626.59705-1ubuntu0.3) ...
    Selecting previously unselected package libwoff1:amd64.
    Preparing to unpack .../11-libwoff1_1.0.2-1build4_amd64.deb ...
    Unpacking libwoff1:amd64 (1.0.2-1build4) ...
    Selecting previously unselected package dvisvgm.
    Preparing to unpack .../12-dvisvgm_2.13.1-1_amd64.deb ...
    Unpacking dvisvgm (2.13.1-1) ...
    Selecting previously unselected package fonts-lmodern.
    Preparing to unpack .../13-fonts-lmodern_2.004.5-6.1_all.deb ...
    Unpacking fonts-lmodern (2.004.5-6.1) ...
    Selecting previously unselected package fonts-noto-mono.
    Preparing to unpack .../14-fonts-noto-mono_20201225-1build1_all.deb ...
    Unpacking fonts-noto-mono (20201225-1build1) ...
    Selecting previously unselected package fonts-texgyre.
    Preparing to unpack .../15-fonts-texgyre_20180621-3.1_all.deb ...
    Unpacking fonts-texgyre (20180621-3.1) ...
    Selecting previously unselected package libapache-pom-java.
    Preparing to unpack .../16-libapache-pom-java_18-1_all.deb ...
    Unpacking libapache-pom-java (18-1) ...
    Selecting previously unselected package libcommons-parent-java.
    Preparing to unpack .../17-libcommons-parent-java_43-1_all.deb ...
    Unpacking libcommons-parent-java (43-1) ...
    Selecting previously unselected package libcommons-logging-java.
    Preparing to unpack .../18-libcommons-logging-java_1.2-2_all.deb ...
    Unpacking libcommons-logging-java (1.2-2) ...
    Selecting previously unselected package libptexenc1:amd64.
    Preparing to unpack .../19-libptexenc1_2021.20210626.59705-1ubuntu0.3_amd64.deb ...
    Unpacking libptexenc1:amd64 (2021.20210626.59705-1ubuntu0.3) ...
    Selecting previously unselected package rubygems-integration.
    Preparing to unpack .../20-rubygems-integration_1.18_all.deb ...
    Unpacking rubygems-integration (1.18) ...
    Selecting previously unselected package ruby3.0.
    Preparing to unpack .../21-ruby3.0_3.0.2-7ubuntu2.12_amd64.deb ...
    Unpacking ruby3.0 (3.0.2-7ubuntu2.12) ...
    Selecting previously unselected package ruby-rubygems.
    Preparing to unpack .../22-ruby-rubygems_3.3.5-2ubuntu1.2_all.deb ...
    Unpacking ruby-rubygems (3.3.5-2ubuntu1.2) ...
    Selecting previously unselected package ruby.
    Preparing to unpack .../23-ruby_1%3a3.0~exp1_amd64.deb ...
    Unpacking ruby (1:3.0~exp1) ...
    Selecting previously unselected package rake.
    Preparing to unpack .../24-rake_13.0.6-2_all.deb ...
    Unpacking rake (13.0.6-2) ...
    Selecting previously unselected package ruby-net-telnet.
    Preparing to unpack .../25-ruby-net-telnet_0.1.1-2_all.deb ...
    Unpacking ruby-net-telnet (0.1.1-2) ...
    Selecting previously unselected package ruby-webrick.
    Preparing to unpack .../26-ruby-webrick_1.7.0-3ubuntu0.2_all.deb ...
    Unpacking ruby-webrick (1.7.0-3ubuntu0.2) ...
    Selecting previously unselected package ruby-xmlrpc.
    Preparing to unpack .../27-ruby-xmlrpc_0.3.2-1ubuntu0.1_all.deb ...
    Unpacking ruby-xmlrpc (0.3.2-1ubuntu0.1) ...
    Selecting previously unselected package libruby3.0:amd64.
    Preparing to unpack .../28-libruby3.0_3.0.2-7ubuntu2.12_amd64.deb ...
    Unpacking libruby3.0:amd64 (3.0.2-7ubuntu2.12) ...
    Selecting previously unselected package libsynctex2:amd64.
    Preparing to unpack .../29-libsynctex2_2021.20210626.59705-1ubuntu0.3_amd64.deb ...
    Unpacking libsynctex2:amd64 (2021.20210626.59705-1ubuntu0.3) ...
    Selecting previously unselected package libteckit0:amd64.
    Preparing to unpack .../30-libteckit0_2.5.11+ds1-1_amd64.deb ...
    Unpacking libteckit0:amd64 (2.5.11+ds1-1) ...
    Selecting previously unselected package libtexlua53:amd64.
    Preparing to unpack .../31-libtexlua53_2021.20210626.59705-1ubuntu0.3_amd64.deb ...
    Unpacking libtexlua53:amd64 (2021.20210626.59705-1ubuntu0.3) ...
    Selecting previously unselected package libtexluajit2:amd64.
    Preparing to unpack .../32-libtexluajit2_2021.20210626.59705-1ubuntu0.3_amd64.deb ...
    Unpacking libtexluajit2:amd64 (2021.20210626.59705-1ubuntu0.3) ...
    Selecting previously unselected package libzzip-0-13:amd64.
    Preparing to unpack .../33-libzzip-0-13_0.13.72+dfsg.1-1.1_amd64.deb ...
    Unpacking libzzip-0-13:amd64 (0.13.72+dfsg.1-1.1) ...
    Selecting previously unselected package xfonts-encodings.
    Preparing to unpack .../34-xfonts-encodings_1%3a1.0.5-0ubuntu2_all.deb ...
    Unpacking xfonts-encodings (1:1.0.5-0ubuntu2) ...
    Selecting previously unselected package xfonts-utils.
    Preparing to unpack .../35-xfonts-utils_1%3a7.7+6build2_amd64.deb ...
    Unpacking xfonts-utils (1:7.7+6build2) ...
    Selecting previously unselected package lmodern.
    Preparing to unpack .../36-lmodern_2.004.5-6.1_all.deb ...
    Unpacking lmodern (2.004.5-6.1) ...
    Selecting previously unselected package preview-latex-style.
    Preparing to unpack .../37-preview-latex-style_12.2-1ubuntu1_all.deb ...
    Unpacking preview-latex-style (12.2-1ubuntu1) ...
    Selecting previously unselected package t1utils.
    Preparing to unpack .../38-t1utils_1.41-4build2_amd64.deb ...
    Unpacking t1utils (1.41-4build2) ...
    Selecting previously unselected package teckit.
    Preparing to unpack .../39-teckit_2.5.11+ds1-1_amd64.deb ...
    Unpacking teckit (2.5.11+ds1-1) ...
    Selecting previously unselected package tex-gyre.
    Preparing to unpack .../40-tex-gyre_20180621-3.1_all.deb ...
    Unpacking tex-gyre (20180621-3.1) ...
    Selecting previously unselected package texlive-binaries.
    Preparing to unpack .../41-texlive-binaries_2021.20210626.59705-1ubuntu0.3_amd64.deb ...
    Unpacking texlive-binaries (2021.20210626.59705-1ubuntu0.3) ...
    Selecting previously unselected package texlive-base.
    Preparing to unpack .../42-texlive-base_2021.20220204-1_all.deb ...
    Unpacking texlive-base (2021.20220204-1) ...
    Selecting previously unselected package texlive-fonts-recommended.
    Preparing to unpack .../43-texlive-fonts-recommended_2021.20220204-1_all.deb ...
    Unpacking texlive-fonts-recommended (2021.20220204-1) ...
    Selecting previously unselected package texlive-latex-base.
    Preparing to unpack .../44-texlive-latex-base_2021.20220204-1_all.deb ...
    Unpacking texlive-latex-base (2021.20220204-1) ...
    Selecting previously unselected package libfontbox-java.
    Preparing to unpack .../45-libfontbox-java_1%3a1.8.16-2_all.deb ...
    Unpacking libfontbox-java (1:1.8.16-2) ...
    Selecting previously unselected package libpdfbox-java.
    Preparing to unpack .../46-libpdfbox-java_1%3a1.8.16-2_all.deb ...
    Unpacking libpdfbox-java (1:1.8.16-2) ...
    Selecting previously unselected package texlive-latex-recommended.
    Preparing to unpack .../47-texlive-latex-recommended_2021.20220204-1_all.deb ...
    Unpacking texlive-latex-recommended (2021.20220204-1) ...
    Selecting previously unselected package texlive-pictures.
    Preparing to unpack .../48-texlive-pictures_2021.20220204-1_all.deb ...
    Unpacking texlive-pictures (2021.20220204-1) ...
    Selecting previously unselected package texlive-latex-extra.
    Preparing to unpack .../49-texlive-latex-extra_2021.20220204-1_all.deb ...
    Unpacking texlive-latex-extra (2021.20220204-1) ...
    Selecting previously unselected package texlive-plain-generic.
    Preparing to unpack .../50-texlive-plain-generic_2021.20220204-1_all.deb ...
    Unpacking texlive-plain-generic (2021.20220204-1) ...
    Selecting previously unselected package tipa.
    Preparing to unpack .../51-tipa_2%3a1.3-21_all.deb ...
    Unpacking tipa (2:1.3-21) ...
    Selecting previously unselected package texlive-xetex.
    Preparing to unpack .../52-texlive-xetex_2021.20220204-1_all.deb ...
    Unpacking texlive-xetex (2021.20220204-1) ...
    Setting up fonts-lato (2.0-2.1) ...
    Setting up fonts-noto-mono (20201225-1build1) ...
    Setting up libwoff1:amd64 (1.0.2-1build4) ...
    Setting up libtexlua53:amd64 (2021.20210626.59705-1ubuntu0.3) ...
    Setting up libijs-0.35:amd64 (0.35-15build2) ...
    Setting up libtexluajit2:amd64 (2021.20210626.59705-1ubuntu0.3) ...
    Setting up libfontbox-java (1:1.8.16-2) ...
    Setting up rubygems-integration (1.18) ...
    Setting up libzzip-0-13:amd64 (0.13.72+dfsg.1-1.1) ...
    Setting up fonts-urw-base35 (20200910-1) ...
    Setting up poppler-data (0.4.11-1) ...
    Setting up tex-common (6.17) ...
    debconf: unable to initialize frontend: Dialog
    debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 78.)
    debconf: falling back to frontend: Readline
    update-language: texlive-base not installed and configured, doing nothing!
    Setting up libjbig2dec0:amd64 (0.19-3build2) ...
    Setting up libteckit0:amd64 (2.5.11+ds1-1) ...
    Setting up libapache-pom-java (18-1) ...
    Setting up ruby-net-telnet (0.1.1-2) ...
    Setting up xfonts-encodings (1:1.0.5-0ubuntu2) ...
    Setting up t1utils (1.41-4build2) ...
    Setting up libidn12:amd64 (1.38-4ubuntu1) ...
    Setting up fonts-texgyre (20180621-3.1) ...
    Setting up libkpathsea6:amd64 (2021.20210626.59705-1ubuntu0.3) ...
    Setting up ruby-webrick (1.7.0-3ubuntu0.2) ...
    Setting up fonts-lmodern (2.004.5-6.1) ...
    Setting up fonts-droid-fallback (1:6.0.1r16-1.1build1) ...
    Setting up ruby-xmlrpc (0.3.2-1ubuntu0.1) ...
    Setting up libsynctex2:amd64 (2021.20210626.59705-1ubuntu0.3) ...
    Setting up libgs9-common (9.55.0~dfsg1-0ubuntu5.13) ...
    Setting up teckit (2.5.11+ds1-1) ...
    Setting up libpdfbox-java (1:1.8.16-2) ...
    Setting up libgs9:amd64 (9.55.0~dfsg1-0ubuntu5.13) ...
    Setting up preview-latex-style (12.2-1ubuntu1) ...
    Setting up libcommons-parent-java (43-1) ...
    Setting up dvisvgm (2.13.1-1) ...
    Setting up libcommons-logging-java (1.2-2) ...
    Setting up xfonts-utils (1:7.7+6build2) ...
    Setting up libptexenc1:amd64 (2021.20210626.59705-1ubuntu0.3) ...
    Setting up texlive-binaries (2021.20210626.59705-1ubuntu0.3) ...
    update-alternatives: using /usr/bin/xdvi-xaw to provide /usr/bin/xdvi.bin (xdvi.bin) in auto mode
    update-alternatives: using /usr/bin/bibtex.original to provide /usr/bin/bibtex (bibtex) in auto mode
    Setting up lmodern (2.004.5-6.1) ...
    Setting up texlive-base (2021.20220204-1) ...
    /usr/bin/ucfr
    /usr/bin/ucfr
    /usr/bin/ucfr
    /usr/bin/ucfr
    tl-paper: setting paper size for dvips to a4: /var/lib/texmf/dvips/config/config-paper.ps
    tl-paper: setting paper size for dvipdfmx to a4: /var/lib/texmf/dvipdfmx/dvipdfmx-paper.cfg
    tl-paper: setting paper size for xdvi to a4: /var/lib/texmf/xdvi/XDvi-paper
    tl-paper: setting paper size for pdftex to a4: /var/lib/texmf/tex/generic/tex-ini-files/pdftexconfig.tex
    debconf: unable to initialize frontend: Dialog
    debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 78.)
    debconf: falling back to frontend: Readline
    Setting up tex-gyre (20180621-3.1) ...
    Setting up texlive-plain-generic (2021.20220204-1) ...
    Setting up texlive-latex-base (2021.20220204-1) ...
    Setting up texlive-latex-recommended (2021.20220204-1) ...
    Setting up texlive-pictures (2021.20220204-1) ...
    Setting up texlive-fonts-recommended (2021.20220204-1) ...
    Setting up tipa (2:1.3-21) ...
    Setting up texlive-latex-extra (2021.20220204-1) ...
    Setting up texlive-xetex (2021.20220204-1) ...
    Setting up rake (13.0.6-2) ...
    Setting up libruby3.0:amd64 (3.0.2-7ubuntu2.12) ...
    Setting up ruby3.0 (3.0.2-7ubuntu2.12) ...
    Setting up ruby (1:3.0~exp1) ...
    Setting up ruby-rubygems (3.3.5-2ubuntu1.2) ...
    Processing triggers for man-db (2.10.2-1) ...
    Processing triggers for mailcap (3.70+nmu1ubuntu1) ...
    Processing triggers for fontconfig (2.13.1-4.2ubuntu5) ...
    Processing triggers for libc-bin (2.35-0ubuntu3.13) ...
    /sbin/ldconfig.real: /usr/local/lib/libur_loader.so.0 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libhwloc.so.15 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libumf.so.1 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libtbb.so.12 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libur_adapter_opencl.so.0 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libtbbbind.so.3 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libtbbmalloc.so.2 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libtbbbind_2_5.so.3 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libtbbmalloc_proxy.so.2 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libur_adapter_level_zero.so.0 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libtcm_debug.so.1 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libtcm.so.1 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libur_adapter_level_zero_v2.so.0 is not a symbolic link
    
    /sbin/ldconfig.real: /usr/local/lib/libtbbbind_2_0.so.3 is not a symbolic link
    
    Processing triggers for tex-common (6.17) ...
    debconf: unable to initialize frontend: Dialog
    debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 78.)
    debconf: falling back to frontend: Readline
    Running updmap-sys. This may take some time... done.
    Running mktexlsr /var/lib/texmf ... done.
    Building format(s) --all.
    	This may take some time... done.
    Reading package lists...
    Building dependency tree...
    Reading state information...
    The following NEW packages will be installed:
      fonts-freefont-ttf
    0 upgraded, 1 newly installed, 0 to remove and 60 not upgraded.
    Need to get 2,388 kB of archives.
    After this operation, 6,653 kB of additional disk space will be used.
    Get:1 http://archive.ubuntu.com/ubuntu jammy/main amd64 fonts-freefont-ttf all 20120503-10build1 [2,388 kB]
    Fetched 2,388 kB in 2s (1,058 kB/s)
    Selecting previously unselected package fonts-freefont-ttf.
    (Reading database ... (Reading database ... 5%(Reading database ... 10%(Reading database ... 15%(Reading database ... 20%(Reading database ... 25%(Reading database ... 30%(Reading database ... 35%(Reading database ... 40%(Reading database ... 45%(Reading database ... 50%(Reading database ... 55%(Reading database ... 60%(Reading database ... 65%(Reading database ... 70%(Reading database ... 75%(Reading database ... 80%(Reading database ... 85%(Reading database ... 90%(Reading database ... 95%(Reading database ... 100%(Reading database ... 155020 files and directories currently installed.)
    Preparing to unpack .../fonts-freefont-ttf_20120503-10build1_all.deb ...
    Unpacking fonts-freefont-ttf (20120503-10build1) ...
    Setting up fonts-freefont-ttf (20120503-10build1) ...
    Processing triggers for fontconfig (2.13.1-4.2ubuntu5) ...


    W: Skipping acquire of configured file 'main/source/Sources' as repository 'https://r2u.stat.illinois.edu/ubuntu jammy InRelease' does not seem to provide it (sources.list entry misspelt?)
    W: Failed to fetch https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu/dists/jammy/InRelease  Could not connect to ppa.launchpadcontent.net:443 (185.125.190.80), connection timed out
    W: Failed to fetch https://ppa.launchpadcontent.net/ubuntugis/ppa/ubuntu/dists/jammy/InRelease  Unable to connect to ppa.launchpadcontent.net:443:
    W: Some index files failed to download. They have been ignored, or old ones used instead.
    debconf: unable to initialize frontend: Dialog
    debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 78, <> line 53.)
    debconf: falling back to frontend: Readline
    debconf: unable to initialize frontend: Readline
    debconf: (This frontend requires a controlling tty.)
    debconf: falling back to frontend: Teletype
    dpkg-preconfigure: unable to re-open stdin: 
    debconf: unable to initialize frontend: Dialog
    debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 78, <> line 1.)
    debconf: falling back to frontend: Readline
    debconf: unable to initialize frontend: Readline
    debconf: (This frontend requires a controlling tty.)
    debconf: falling back to frontend: Teletype
    dpkg-preconfigure: unable to re-open stdin: 



```python
!python3 Distribution_Estimation/get_plots.py --add_consts 0.003 0.005 0.007 --discounts 0.5 0.6 0.7
```

    [nltk_data] Downloading package punkt_tab to /root/nltk_data...
    [nltk_data]   Package punkt_tab is already up-to-date!
    
    ===== Running for glove =====
    Loading GloVe (100d)...
    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    Tokenizing train corpus...
    Tokenizing test corpus...
    train size : 100000
    test size : 10000
    Building cache for glove...
    Vocabulary size: 66753
    ===============Sanity Check=============
    the 25545.929466294947
    a 14715.220579368786
    from 9945.499147808556
    16363
    149552
    Vocab size: 66753
    Top words: ['cleverest', 'editions', 'mogadiscio', 'fabulously', 'animated', 'krause', 'capehart', 'weeded', 'agut', 'doofenshmirthz', 'reads', 'flautist', 'busy', 'lsbeck', 'gliding', 'alun', 'polymeric', 'binder', 'eckholdt', 'abruptly']
    Max L1 norm: 4.0666
    Word with max norm: republish
    13139
    0 cleverest [('husbandman', 0.004879800137132406), ('xion', 0.004843533504754305), ('maneater', 0.004802011419087648), ('monsen', 0.004728770814836025), ('parasaurolophus', 0.004633008036762476), ('ragale', 0.004632826894521713), ('ironically', 0.004620539490133524), ('trimurti', 0.0046097333543002605), ('abusir', 0.0045907385647296906), ('harihara', 0.004510394763201475), ('megaliths', 0.004506304394453764), ('arguably', 0.0044969357550144196), ('bajoran', 0.004484734032303095), ('arihant', 0.004478070419281721), ('berardi', 0.004477867856621742), ('clich', 0.004470181185752153), ('machismo', 0.004464300815016031), ('dags', 0.0044519719667732716), ('dodos', 0.004451197572052479), ('rarity', 0.004450962413102388), ('kantara', 0.004429037682712078), ('morhange', 0.004424056503921747), ('characterisation', 0.004412782844156027), ('inocybe', 0.004389320500195026), ('jallon', 0.004376937169581652), ('rediscovered', 0.004371533170342445), ('hadrosaurids', 0.0043605477549135685), ('chapterhouse', 0.004345564637333155), ('veerashaiva', 0.004335375968366861), ('gabbar', 0.00433410843834281), ('djedkare', 0.004332135897129774), ('divinities', 0.004325838293880224), ('narvesen', 0.004322135355323553), ('bessin', 0.004318945109844208), ('dhangar', 0.004311538301408291), ('immortals', 0.004311248194426298), ('andorian', 0.004307042807340622), ('pfaster', 0.004306319169700146), ('berio', 0.004306027200073004), ('westernport', 0.004304933361709118), ('aforementioned', 0.004299265332520008), ('batou', 0.0042904638685286045), ('typography', 0.0042780195362865925), ('eucalypts', 0.004273644182831049), ('leviathan', 0.004272029269486666), ('muscimol', 0.004269585013389587), ('raghuveer', 0.004265669733285904), ('toirdelbach', 0.0042608496733009815), ('wherein', 0.004260081332176924), ('entertainer', 0.004252731334418058)]
    20000 porvenir [('abusir', 0.005060700234025717), ('toso', 0.004990996792912483), ('westernport', 0.004946856293827295), ('kantara', 0.004941090941429138), ('ragale', 0.004760073032230139), ('husbandman', 0.0047486103139817715), ('parasaurolophus', 0.004731377121061087), ('monsen', 0.004710403736680746), ('mondlane', 0.00470900209620595), ('eshmun', 0.0045209722593426704), ('steeltown', 0.0045167081989347935), ('taneytown', 0.004491182044148445), ('bohemica', 0.0044859424233436584), ('bessin', 0.004475994501262903), ('dags', 0.0044744787737727165), ('xion', 0.0044147721491754055), ('werneth', 0.004407811444252729), ('palestro', 0.004403611179441214), ('andorian', 0.004396142903715372), ('morhange', 0.004391846247017384), ('inocybe', 0.004390207584947348), ('mentmore', 0.004380233585834503), ('leukemic', 0.004379502031952143), ('chapterhouse', 0.0043746051378548145), ('mcgary', 0.004371486604213715), ('bosi', 0.004371446091681719), ('arihant', 0.004369971342384815), ('hubbardton', 0.0043692803010344505), ('cytogenetics', 0.004358802922070026), ('varrick', 0.00435150321573019), ('bsu', 0.0043428558856248856), ('djedkare', 0.0043379501439630985), ('arromanches', 0.004325838293880224), ('corythosaurus', 0.004320288076996803), ('berardi', 0.004319390747696161), ('palenque', 0.004319264553487301), ('dareus', 0.004302367102354765), ('lonaconing', 0.004298399668186903), ('mithra', 0.004288979806005955), ('maneater', 0.004285428673028946), ('biddenden', 0.004277982749044895), ('geastrum', 0.004277766682207584), ('toirdelbach', 0.004269296303391457), ('agno', 0.0042677223682403564), ('pfaster', 0.004257751628756523), ('muscarine', 0.0042525422759354115), ('dhangar', 0.004252161365002394), ('jallon', 0.004249795340001583), ('valkyria', 0.004231978207826614), ('ardan', 0.004213211592286825)]
    30000 hum [('hum', 0.18689458072185516), ('encore', 0.004221292678266764), ('delight', 0.004038228187710047), ('chapterhouse', 0.00396342808380723), ('xion', 0.003942967392504215), ('cry', 0.003926446195691824), ('flux', 0.003914428874850273), ('ko', 0.003913463559001684), ('shine', 0.003905730787664652), ('io', 0.003870939603075385), ('atkinson', 0.003860367927700281), ('joyful', 0.0038584687281399965), ('dags', 0.0038542277179658413), ('marlene', 0.003847641870379448), ('noises', 0.0038444441743195057), ('interlude', 0.0038304717745631933), ('toso', 0.00382420071400702), ('phantom', 0.0038241015281528234), ('static', 0.0038215946406126022), ('husbandman', 0.003820369951426983), ('shadows', 0.003813000861555338), ('entertainer', 0.0038084390107542276), ('rage', 0.0038027972914278507), ('dd', 0.003785961540415883), ('boadicea', 0.0037851908709853888), ('arihant', 0.0037845815531909466), ('reprised', 0.0037768417969346046), ('er', 0.0037683923728764057), ('bessin', 0.003762125037610531), ('nowhere', 0.0037616845220327377), ('perfection', 0.0037475302815437317), ('mithra', 0.0037463263142853975), ('parasaurolophus', 0.003743697889149189), ('invincible', 0.0037298526149243116), ('cam', 0.0037277902010828257), ('opined', 0.0037224148400127888), ('extremes', 0.0037208956200629473), ('trimurti', 0.003718961961567402), ('bohemica', 0.0037176639307290316), ('jig', 0.0037166655529290438), ('kashi', 0.003715060418471694), ('kantara', 0.0037137300241738558), ('ke', 0.003701545763760805), ('sabor', 0.0036933596711605787), ('tap', 0.0036876939702779055), ('hi', 0.003687571967020631), ('thee', 0.0036855752114206553), ('fab', 0.0036822082474827766), ('na', 0.0036782955285161734), ('taneytown', 0.0036776692140847445)]
    40000 analyzing [('discovering', 0.005637032445520163), ('examining', 0.0055731250904500484), ('surveying', 0.005325187463313341), ('comparing', 0.005269348155707121), ('uncovering', 0.005115286447107792), ('analyzed', 0.005091259256005287), ('reviewing', 0.004911363124847412), ('precisely', 0.004797042813152075), ('identifying', 0.004784146789461374), ('researched', 0.00471229013055563), ('examines', 0.00461203046143055), ('acknowledging', 0.004593056160956621), ('examine', 0.0045744068920612335), ('detailing', 0.0045420583337545395), ('translating', 0.004496385809034109), ('focusing', 0.004458656534552574), ('explaining', 0.0044500757940113544), ('altering', 0.004440661519765854), ('utilizing', 0.004421422258019447), ('confirming', 0.004407763481140137), ('relates', 0.00438673747703433), ('highlighting', 0.004385836888104677), ('applying', 0.004377903416752815), ('studying', 0.0043534040451049805), ('examined', 0.004345819354057312), ('describing', 0.004335414152592421), ('likened', 0.0043342807330191135), ('classify', 0.004294528625905514), ('picking', 0.004282196518033743), ('manipulating', 0.0042702266946434975), ('assess', 0.0042489878833293915), ('combining', 0.004237899091094732), ('calculations', 0.00422713765874505), ('compare', 0.004217405337840319), ('impressions', 0.004209483042359352), ('analyses', 0.004193889908492565), ('proving', 0.0041894931346178055), ('traced', 0.004165454767644405), ('contents', 0.004164489917457104), ('observing', 0.004163092002272606), ('determining', 0.004160952288657427), ('cytogenetics', 0.004144289530813694), ('microscopic', 0.004138772841542959), ('relate', 0.004136607050895691), ('sampling', 0.004128372296690941), ('asserting', 0.004125535953789949), ('assuming', 0.0041231438517570496), ('conclusions', 0.004112069960683584), ('surveys', 0.004101837053894997), ('shifting', 0.0041007488034665585)]
    50000 funeral [('funeral', 0.2740449905395508), ('hometown', 0.004275296349078417), ('rites', 0.0042329393327236176), ('burial', 0.004170878790318966), ('prayers', 0.004157393239438534), ('ceremonies', 0.004102082457393408), ('coffin', 0.0040315622463822365), ('wreath', 0.0039872098714113235), ('solemn', 0.003924141637980938), ('bride', 0.003908335696905851), ('beloved', 0.0038978049997240305), ('ceremony', 0.0038917867932468653), ('mourning', 0.0038807883393019438), ('diana', 0.003875584341585636), ('sermon', 0.003849519183859229), ('burials', 0.0038234402891248465), ('midnight', 0.003791218623518944), ('gathering', 0.003752821357920766), ('celebrations', 0.003725704038515687), ('tribute', 0.0036977045238018036), ('widow', 0.003694998798891902), ('aunt', 0.0036840694956481457), ('wedding', 0.0036791441962122917), ('commemorated', 0.0036669597029685974), ('celebration', 0.0036636171862483025), ('gathered', 0.0036462019197642803), ('relatives', 0.0036414938513189554), ('celebrated', 0.00364023563452065), ('greeted', 0.003635951317846775), ('snyder', 0.003630454884842038), ('memorials', 0.003622939344495535), ('cohen', 0.003621544921770692), ('evening', 0.0036058281548321247), ('accompanied', 0.0035999438259750605), ('occasion', 0.003591485321521759), ('respects', 0.0035849171690642834), ('vengeance', 0.0035808696411550045), ('inauguration', 0.0035733338445425034), ('absent', 0.0035681710578501225), ('hillside', 0.0035635537933558226), ('remembered', 0.0035635344684123993), ('pilgrimage', 0.003560225246474147), ('blessing', 0.003559780539944768), ('parade', 0.003554070834070444), ('eve', 0.0035518633667379618), ('arrival', 0.0035515427589416504), ('deceased', 0.0035387862008064985), ('flock', 0.003534938208758831), ('preceded', 0.003534307237714529), ('uncle', 0.00353131047450006)]
    60000 myrtaceae [('daisy', 0.0033946363255381584), ('aunt', 0.003362284740433097), ('atkinson', 0.0033546159975230694), ('rowson', 0.0033496913965791464), ('husbandman', 0.003330622101202607), ('denise', 0.0033259897027164698), ('heather', 0.003313126042485237), ('abusir', 0.003292107954621315), ('taneytown', 0.00328025221824646), ('morton', 0.0032651606015861034), ('wilcox', 0.0032630872447043657), ('mcgary', 0.0032606013119220734), ('westernport', 0.0032596252858638763), ('arihant', 0.003255656221881509), ('ragale', 0.003235564334318042), ('toso', 0.0032315319404006004), ('prefers', 0.003229232504963875), ('parasaurolophus', 0.003215410513803363), ('alden', 0.0032116735819727182), ('myles', 0.0032056078780442476), ('sara', 0.0031978015322238207), ('eucalypts', 0.0031948161777108908), ('dilke', 0.003193052252754569), ('homarus', 0.00319201429374516), ('berardi', 0.0031918268650770187), ('bsu', 0.003189056646078825), ('conifers', 0.0031882862094789743), ('woodward', 0.0031878252048045397), ('mcdonald', 0.0031857553403824568), ('fulton', 0.00318131479434669), ('reminds', 0.0031805813778191805), ('guez', 0.0031646639108657837), ('lauderdale', 0.0031635083723813295), ('trudy', 0.0031634897459298372), ('bevil', 0.0031615979969501495), ('walton', 0.0031606201082468033), ('arromanches', 0.0031533401925116777), ('mortimer', 0.003145638620480895), ('juliet', 0.003144790418446064), ('oahu', 0.0031382215674966574), ('peck', 0.00313704670406878), ('xion', 0.0031352066434919834), ('neglect', 0.00313473935239017), ('kerr', 0.003132354700937867), ('bessin', 0.0031306801829487085), ('leukemic', 0.0031292119529098272), ('mayo', 0.0031218943186104298), ('mycena', 0.003118709195405245), ('frog', 0.0031167424749583006), ('weevil', 0.003116265870630741)]
    Saved cache → /content/Distribution_Estimation/cache/glove_cache.pkl
    Evaluating Additive Smoothing...
      Add = 0.003
    100% 7/7 [00:52<00:00,  7.57s/it]
      Add = 0.005
    100% 7/7 [00:52<00:00,  7.57s/it]
      Add = 0.007
    100% 7/7 [00:53<00:00,  7.61s/it]
    Evaluating Kneser-Ney...
      Discount = 0.5
    100% 7/7 [01:45<00:00, 15.09s/it]
      Discount = 0.6
    100% 7/7 [01:45<00:00, 15.13s/it]
      Discount = 0.7
    100% 7/7 [01:47<00:00, 15.34s/it]
    
    ===== Running for word2vec =====
    Loading pretrained Word2Vec (Google News 300d)...
    Tokenizing train corpus...
    Tokenizing test corpus...
    train size : 100000
    test size : 10000
    Building cache for word2vec...
    Vocabulary size: 66753
    ===============Sanity Check=============
    the 25545.929466294947
    a 14715.220579368786
    from 9945.499147808556
    16363
    149552
    Vocab size: 66753
    Top words: ['cleverest', 'editions', 'mogadiscio', 'fabulously', 'animated', 'krause', 'capehart', 'weeded', 'agut', 'doofenshmirthz', 'reads', 'flautist', 'busy', 'lsbeck', 'gliding', 'alun', 'polymeric', 'binder', 'eckholdt', 'abruptly']
    Max L1 norm: 4.1875
    Word with max norm: material_objectionable
    12055
    0 cleverest [('clever', 0.004300698172301054), ('gemma', 0.0037660710513591766), ('myles', 0.003765821224078536), ('corbett', 0.0037577482871711254), ('kerr', 0.0037400329019874334), ('savoy', 0.003723734524101019), ('alexandra', 0.0037115158047527075), ('cheltenham', 0.003706068731844425), ('petersburg', 0.0036981760058552027), ('bachchan', 0.003697834210470319), ('morton', 0.0036959710996598005), ('roberto', 0.0036800275556743145), ('emerson', 0.0036783830728381872), ('iain', 0.0036760654766112566), ('dominic', 0.0036600471939891577), ('juliet', 0.003659214125946164), ('presbyterian', 0.0036156990099698305), ('atkinson', 0.0036131602246314287), ('warwick', 0.0036082142032682896), ('armenian', 0.0036005389411002398), ('arguably', 0.0035990800242871046), ('michel', 0.003590955398976803), ('cunningham', 0.003584605874493718), ('romanesque', 0.00358204310759902), ('goldman', 0.0035813164431601763), ('cheshire', 0.003580115968361497), ('avon', 0.0035742856562137604), ('nigel', 0.003574018133804202), ('subgenus', 0.0035682707093656063), ('burt', 0.003567214822396636), ('pamela', 0.0035647121258080006), ('mohamed', 0.003555100644007325), ('barron', 0.0035496815107762814), ('steeltown', 0.003546222113072872), ('luther', 0.003545197658240795), ('sebastian', 0.0035446572583168745), ('thomson', 0.0035316171124577522), ('garrett', 0.003531605238094926), ('ernie', 0.0035287921782583), ('templar', 0.0035266762133687735), ('judaism', 0.003526116255670786), ('andreas', 0.0035207560285925865), ('sergio', 0.003514717100188136), ('peggy', 0.0035073114559054375), ('greenland', 0.0035068087745457888), ('bsu', 0.003495495766401291), ('halifax', 0.0034939702600240707), ('denise', 0.0034876149147748947), ('ibrahim', 0.0034870717208832502), ('lauderdale', 0.003476591082289815)]
    30000 hum [('hum', 0.18689458072185516), ('emerson', 0.003602977842092514), ('juliet', 0.003602105425670743), ('roberto', 0.003595079993829131), ('warwick', 0.0035603989381343126), ('spector', 0.003549594897776842), ('corbett', 0.0035473653115332127), ('atkinson', 0.0035362886264920235), ('bsu', 0.003529235254973173), ('myles', 0.003528488799929619), ('noises', 0.0035235935356467962), ('judaism', 0.0035149131435900927), ('morton', 0.0035076120402663946), ('gemma', 0.0035025030374526978), ('greenland', 0.0034998287446796894), ('burt', 0.003499353537335992), ('ernie', 0.0034959253389388323), ('nha', 0.003478447673842311), ('plymouth', 0.0034753107465803623), ('steeltown', 0.003468461334705353), ('sebastian', 0.003466732567176223), ('cheltenham', 0.003465099725872278), ('avon', 0.0034644578117877245), ('guez', 0.0034637856297194958), ('bachchan', 0.0034635586198419333), ('iain', 0.003461550222709775), ('norfolk', 0.0034564752131700516), ('cunningham', 0.003453647019341588), ('alexandra', 0.0034508726093918085), ('kerr', 0.0034497224260121584), ('din', 0.003442893736064434), ('garrett', 0.0034379709977656603), ('dard', 0.003422281239181757), ('savoy', 0.003417124506086111), ('luther', 0.00341627630405128), ('rutgers', 0.00341366627253592), ('romanesque', 0.003411482321098447), ('petersburg', 0.003409054595977068), ('pola', 0.0034012312535196543), ('noisy', 0.0033934202510863543), ('sabor', 0.003391681704670191), ('goldman', 0.00338434218429029), ('rafael', 0.003379716305062175), ('nol', 0.0033750636503100395), ('andreas', 0.0033750461880117655), ('metallica', 0.0033750229049474), ('fulton', 0.0033655688166618347), ('ipc', 0.0033588530495762825), ('dags', 0.003343696938827634), ('sergio', 0.0033418030943721533)]
    40000 analyzing [('examining', 0.00467011658474803), ('reviewing', 0.004441727884113789), ('analyzed', 0.004276386462152004), ('surveying', 0.004275097046047449), ('corbett', 0.004178927280008793), ('morton', 0.004136569332331419), ('iain', 0.004130607936531305), ('myles', 0.004087897948920727), ('discovering', 0.004084359854459763), ('gemma', 0.004070175811648369), ('recognizing', 0.004042007494717836), ('petersburg', 0.004036640748381615), ('acknowledging', 0.004024572670459747), ('greenland', 0.003993862774223089), ('alexandra', 0.003955422434955835), ('discussing', 0.003945443779230118), ('examine', 0.003943737596273422), ('confirming', 0.0039412169717252254), ('lauderdale', 0.003937332425266504), ('identifying', 0.003935794346034527), ('andreas', 0.00393564673140645), ('atkinson', 0.00392976263538003), ('steeltown', 0.0039029703475534916), ('roberto', 0.0038950773887336254), ('comparing', 0.0038947374559938908), ('cheltenham', 0.0038928331341594458), ('uncovering', 0.0038918890058994293), ('emerson', 0.003891733242198825), ('juliet', 0.0038909248542040586), ('bachchan', 0.0038792011328041553), ('basel', 0.0038753969129174948), ('dominic', 0.00386818521656096), ('cunningham', 0.003868133993819356), ('cns', 0.00386803993023932), ('guez', 0.003860011463984847), ('sergio', 0.003859663149341941), ('indu', 0.003859270364046097), ('analysis', 0.0038590773474425077), ('kerr', 0.003849885892122984), ('subgenus', 0.0038496009074151516), ('halifax', 0.00384921976365149), ('managing', 0.0038443845696747303), ('assuming', 0.003839536802843213), ('observing', 0.0038360643666237593), ('berkshire', 0.003834889503195882), ('luther', 0.0038276880513876677), ('sebastian', 0.0038224898744374514), ('namely', 0.0038219084963202477), ('avon', 0.003821432823315263), ('mohamed', 0.0038118495140224695)]
    50000 funeral [('funeral', 0.2740449905395508), ('burial', 0.0035882238298654556), ('corbett', 0.003391109872609377), ('myles', 0.0033629126846790314), ('petersburg', 0.003336923662573099), ('kerr', 0.003325688187032938), ('guez', 0.0033108533825725317), ('steeltown', 0.0032934704795479774), ('bachchan', 0.00328467832878232), ('gemma', 0.0032821751665323973), ('iain', 0.0032735317945480347), ('morton', 0.003270126646384597), ('emerson', 0.0032580606639385223), ('avon', 0.0032457862980663776), ('memorial', 0.0032424533274024725), ('lauderdale', 0.0032408274710178375), ('greenland', 0.003234370844438672), ('burt', 0.0032263847533613443), ('romanesque', 0.003217579098418355), ('dominic', 0.003216072218492627), ('roberto', 0.003213129937648773), ('sergio', 0.0032020986545830965), ('presbyterian', 0.0032014423049986362), ('bsu', 0.0032001687213778496), ('warwick', 0.003195200115442276), ('mohamed', 0.003194470191374421), ('juliet', 0.0031864154152572155), ('alexandra', 0.003184855217114091), ('langer', 0.003182494780048728), ('cheltenham', 0.00317935342900455), ('ibrahim', 0.0031790544744580984), ('bride', 0.0031782796140760183), ('halifax', 0.0031749894842505455), ('burials', 0.0031666557770222425), ('savoy', 0.0031639907974749804), ('coffin', 0.0031593504827469587), ('berkshire', 0.003149268217384815), ('rites', 0.003142312867566943), ('lanka', 0.003141699591651559), ('atkinson', 0.003138488158583641), ('sebastian', 0.0031322103459388018), ('judaism', 0.0031284645665436983), ('stipe', 0.003124175826087594), ('bury', 0.0031216186471283436), ('jewell', 0.003120262874290347), ('rafael', 0.0031200896482914686), ('templar', 0.0031183231621980667), ('michel', 0.0031170635484158993), ('memorials', 0.0031166758853942156), ('luther', 0.003116307547315955)]
    Saved cache → /content/Distribution_Estimation/cache/word2vec_cache.pkl
    Evaluating Additive Smoothing...
      Add = 0.003
    100% 7/7 [00:35<00:00,  5.10s/it]
      Add = 0.005
    100% 7/7 [00:36<00:00,  5.28s/it]
      Add = 0.007
    100% 7/7 [00:34<00:00,  4.93s/it]
    Evaluating Kneser-Ney...
      Discount = 0.5
    100% 7/7 [01:16<00:00, 10.96s/it]
      Discount = 0.6
    100% 7/7 [01:11<00:00, 10.24s/it]
      Discount = 0.7
    100% 7/7 [01:13<00:00, 10.50s/it]
    
    ===== Running for gpt2 =====
    Loading GPT-2 embeddings (768d)...
    Loading weights: 100% 148/148 [00:00<00:00, 355.12it/s, Materializing param=wte.weight]
    [1mGPT2Model LOAD REPORT[0m from: gpt2
    Key                  | Status     |  | 
    ---------------------+------------+--+-
    h.{0...11}.attn.bias | [38;5;208mUNEXPECTED[0m |  | 
    
    [3mNotes:
    - [38;5;208mUNEXPECTED[0m[3m	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.[0m
    Tokenizing train corpus...
    Tokenizing test corpus...
    train size : 100000
    test size : 10000
    Building cache for gpt2...
    Vocabulary size: 26810
    ===============Sanity Check=============
    the 8388.843816872617
    a 5533.090991940313
    from 185.52082594565036
    3636
    17004
    Vocab size: 26810
    Top words: ['Ġimpression', 'Ġwear', 'Ġkidnap', 'reads', 'Ġrefresh', 'Ġpaternity', 'Ġslides', 'Ġmid', 'Ġwant', 'Ġimplications', 'Ġtechn', 'Ġreneg', 'abeth', 'duc', 'Ġriot', 'inv', 'Ġexport', 'Ġaugment', 'equ', 'Ġvarieties']
    Max L1 norm: 1.7851565
    Word with max norm: ,
    14284
    0 Ġimpression [('Ġimpression', 1.820448637008667), ('Ġimpressions', 0.0026199098210781813), ('Ġindication', 0.0020373743027448654), ('Ġperception', 0.0020367924589663744), ('Ġillusion', 0.00199507107026875), ('Ġawkward', 0.0019817634020000696), ('Ġsensation', 0.0019549583084881306), ('Ġexceptionally', 0.0019463903736323118), ('Ġnotion', 0.0019401800818741322), ('Ġoutspoken', 0.0019385699415579438), ('Ġannoyed', 0.0019286395981907845), ('Ġfrustrating', 0.001921305782161653), ('Ġinterpretations', 0.0019067254615947604), ('Ġexcitement', 0.0019016433507204056), ('Ġcriticisms', 0.0018994222627952695), ('Ġportrayal', 0.0018953203689306974), ('Ġdiscovering', 0.001893861684948206), ('Ġhardcore', 0.0018887328915297985), ('Ġdismay', 0.0018884502351284027), ('Ġfake', 0.0018880150746554136), ('Ġcute', 0.0018856446258723736), ('Ġdelight', 0.0018851321656256914), ('Ġinterpretation', 0.001884102588519454), ('Ġdemeanor', 0.001881469041109085), ('Ġinsight', 0.0018802048871293664), ('Ġevaluation', 0.001878812676295638), ('Ġnaive', 0.0018774436321109533), ('Ġfindings', 0.001876574708148837), ('Ġsuggestions', 0.0018764673732221127), ('Ġvibe', 0.0018762145191431046), ('Ġpsychological', 0.0018761251121759415), ('Ġcautious', 0.0018743983237072825), ('Ġinstinct', 0.0018739164806902409), ('Ġconfused', 0.0018735813209787011), ('Ġexpertise', 0.0018721319502219558), ('Ġassumption', 0.0018687868723645806), ('Ġshocked', 0.0018673938466235995), ('Ġpleasant', 0.0018656182801350951), ('Ġthinks', 0.0018636789172887802), ('Ġcomputational', 0.0018632959108799696), ('Ġstylish', 0.0018611227860674262), ('Ġoutraged', 0.0018593382555991411), ('Ġskeptical', 0.0018591260304674506), ('Ġemergence', 0.00185879273340106), ('Ġargument', 0.001858611823990941), ('Ġincredible', 0.0018584373174235225), ('Ġemotion', 0.0018569864332675934), ('Ġideology', 0.0018551346147432923), ('Ġaspect', 0.0018548652296885848), ('Ġoutline', 0.0018542986363172531)]
    10000 Ġconject [('Ġspeculated', 0.0021204077638685703), ('Ġcomputational', 0.0019650321919471025), ('Ġhypothesis', 0.0019550370052456856), ('Ġsuggestions', 0.0019452610285952687), ('Ġdescriptive', 0.0019395753042772412), ('Ġconclusions', 0.0019361457088962197), ('Ġcriticisms', 0.0019327232148498297), ('Ġmathematics', 0.0019269305048510432), ('Ġdispar', 0.0019182813121005893), ('Ġasserts', 0.0019175315974280238), ('Ġinterpretations', 0.0019170987652614713), ('Ġcontradicted', 0.001913151703774929), ('Ġdoctrines', 0.0019092822913080454), ('Ġjumped', 0.0019081649370491505), ('Ġdesires', 0.0019078857731074095), ('Ġexceptionally', 0.0019072400173172355), ('Ġrumors', 0.001906034303829074), ('Ġremark', 0.0019055079901590943), ('Ġspurred', 0.0019054991425946355), ('Ġskeptical', 0.001902576768770814), ('Ġimagined', 0.001896691625006497), ('Ġimplies', 0.0018963919719681144), ('Ġded', 0.0018960178131237626), ('Ġexamines', 0.00189531862270087), ('Ġfindings', 0.0018936552805826068), ('Ġfrustrating', 0.0018931110389530659), ('Ġreconstruct', 0.0018913389649242163), ('Ġprecept', 0.001890168758109212), ('Ġfeasible', 0.0018892221851274371), ('Ġdisagreements', 0.001887166639789939), ('Ġambiguous', 0.0018867373000830412), ('Ġuncovered', 0.0018865953898057342), ('Ġprofess', 0.0018859095871448517), ('Ġtheories', 0.0018848746549338102), ('Ġspeculation', 0.0018845826853066683), ('Ġaccusations', 0.0018819611286744475), ('Ġremarked', 0.0018815341172739863), ('Ġarising', 0.0018809690373018384), ('Ġseeks', 0.0018792947521433234), ('Ġlamented', 0.001878927112556994), ('Ġstrive', 0.0018777251243591309), ('Ġpresumed', 0.0018751899478957057), ('Ġpropose', 0.00187512650154531), ('Ġphilosophers', 0.0018749802839010954), ('Ġlament', 0.0018739799270406365), ('Ġdiscovering', 0.0018737745704129338), ('Ġcalculated', 0.0018729916773736477), ('Ġassumed', 0.0018700184300541878), ('Ġcalculations', 0.0018664980307221413), ('Ġdismay', 0.0018662572838366032)]
    20000 Ġcensored [('Ġcensored', 0.6424664855003357), ('Ġcensorship', 0.002610239200294018), ('Ġsuppressed', 0.0021309976000338793), ('Ġcens', 0.0021304518450051546), ('Ġprotested', 0.002099434845149517), ('Ġmandated', 0.0020763978827744722), ('Ġblocked', 0.0020734556019306183), ('Ġforbidden', 0.0020604575984179974), ('Ġoutspoken', 0.0020288664381951094), ('Ġcontradicted', 0.002011641627177596), ('Ġhospitalized', 0.002004185225814581), ('Ġannoyed', 0.001989306416362524), ('Ġoutraged', 0.0019864237401634455), ('Ġdeleted', 0.0019863497000187635), ('Ġslated', 0.001980425789952278), ('Ġcrushed', 0.001970477867871523), ('Ġbiased', 0.00196716096252203), ('Ġfrustrating', 0.001966313924640417), ('Ġjumped', 0.00196489947848022), ('Ġpornographic', 0.0019637385848909616), ('Ġcute', 0.0019625965505838394), ('Ġeroded', 0.0019601972308009863), ('Ġswitched', 0.0019583231769502163), ('Ġspurred', 0.0019537596963346004), ('Ġcriticisms', 0.0019507297547534108), ('Ġrude', 0.0019506494281813502), ('Ġencoded', 0.001947246491909027), ('Ġbombed', 0.0019461145857349038), ('Ġterminated', 0.0019442965276539326), ('Ġedits', 0.0019412152469158173), ('Ġunc', 0.0019408495863899589), ('Ġunconstitutional', 0.0019387973006814718), ('Ġhugely', 0.0019371119560673833), ('Ġcomputational', 0.0019367645727470517), ('Ġbanned', 0.001936202053911984), ('Ġlamented', 0.0019340976141393185), ('Ġdissenting', 0.001933286082930863), ('Ġcirculated', 0.0019326979527249932), ('Ġchooses', 0.0019315795507282019), ('Ġclassify', 0.0019305507885292172), ('Ġimprisoned', 0.001930164871737361), ('Ġbarred', 0.0019293554360046983), ('Ġcrude', 0.0019291636999696493), ('Ġhindered', 0.0019290726631879807), ('Ġcomplained', 0.0019283209694549441), ('Ġpsychiatric', 0.0019273330690339208), ('Ġdismay', 0.0019271797500550747), ('Ġdensely', 0.0019245590083301067), ('Ġdictator', 0.0019239914836362004), ('Ġhardcore', 0.0019235650543123484)]
    Saved cache → /content/Distribution_Estimation/cache/gpt2_cache.pkl
    Evaluating Additive Smoothing...
      Add = 0.003
    100% 7/7 [01:13<00:00, 10.57s/it]
      Add = 0.005
    100% 7/7 [01:13<00:00, 10.51s/it]
      Add = 0.007
    100% 7/7 [01:14<00:00, 10.61s/it]
    Evaluating Kneser-Ney...
      Discount = 0.5
    100% 7/7 [02:21<00:00, 20.21s/it]
      Discount = 0.6
    100% 7/7 [02:22<00:00, 20.41s/it]
      Discount = 0.7
    100% 7/7 [02:22<00:00, 20.40s/it]
    
    ===== ADD-CONSTANT RESULTS =====
    
    --- GLOVE ---
    m | Add=0.003 | Add=0.005 | Add=0.007
    -------------------------------------
     0 |   706.79 |   719.72 |   736.46
     5 |   645.24 |   668.98 |   691.94
    10 |   618.76 |   646.84 |   672.37
    20 |   589.88 |   622.53 |   650.88
    30 |   572.92 |   608.23 |   638.27
    40 |   561.84 |   598.91 |   630.11
    50 |   554.15 |   592.46 |   624.50
    
    --- WORD2VEC ---
    m | Add=0.003 | Add=0.005 | Add=0.007
    -------------------------------------
     0 |   706.79 |   719.72 |   736.46
     5 |   661.18 |   683.83 |   705.93
    10 |   644.95 |   671.17 |   695.35
    20 |   628.43 |   658.38 |   684.87
    30 |   618.99 |   651.30 |   679.31
    40 |   612.73 |   646.73 |   675.85
    50 |   608.66 |   643.96 |   673.94
    
    --- GPT2 ---
    m | Add=0.003 | Add=0.005 | Add=0.007
    -------------------------------------
     0 |   242.68 |   241.90 |   244.16
     5 |   232.52 |   234.27 |   237.87
    10 |   227.01 |   230.12 |   234.47
    20 |   220.34 |   225.05 |   230.30
    30 |   216.07 |   221.77 |   227.60
    40 |   213.02 |   219.43 |   225.69
    50 |   210.33 |   217.35 |   223.99
    
    ===== KNESER-NEY RESULTS =====
    
    --- GLOVE ---
    m | Discount=0.5 | Discount=0.6 | Discount=0.7
    ----------------------------------------------
     0 |   318.83 |   309.77 |   303.65
     5 |   303.06 |   296.80 |   292.86
    10 |   299.59 |   293.83 |   290.27
    20 |   295.57 |   290.43 |   287.36
    30 |   293.12 |   288.41 |   285.67
    40 |   291.63 |   287.23 |   284.74
    50 |   290.54 |   286.40 |   284.11
    
    --- WORD2VEC ---
    m | Discount=0.5 | Discount=0.6 | Discount=0.7
    ----------------------------------------------
     0 |   318.83 |   309.77 |   303.65
     5 |   305.77 |   299.16 |   294.95
    10 |   304.43 |   298.08 |   294.08
    20 |   303.12 |   297.13 |   293.41
    30 |   302.56 |   296.81 |   293.29
    40 |   302.28 |   296.73 |   293.36
    50 |   302.25 |   296.86 |   293.63
    
    --- GPT2 ---
    m | Discount=0.5 | Discount=0.6 | Discount=0.7
    ----------------------------------------------
     0 |   167.56 |   164.15 |   162.10
     5 |   171.96 |   168.68 |   166.75
    10 |   171.25 |   168.11 |   166.27
    20 |   170.26 |   167.31 |   165.61
    30 |   169.58 |   166.77 |   165.19
    40 |   169.09 |   166.40 |   164.90
    50 |   168.50 |   165.93 |   164.53
    
    All experiments complete.



```python

```
