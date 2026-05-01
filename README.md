# Clone the repository
```python
!git clone https://github.com/BaristaBandits/Distribution_Estimation.git
```

# Install the required packages
```python
!pip install -r Distribution_Estimation/requirements.txt
```

# Install the following for plotting in IEEEstle.mplstyle
```bash
%%bash
sudo apt-get update
sudo apt-get install -y texlive-xetex texlive-latex-extra texlive-fonts-recommended
sudo apt-get install -y fonts-freefont-ttf
```

# Run to get results

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
