# Bispectrum

This repository provides simple implementations and demos of the 2-dimensional translation-invariant bispectrum

## Installation

To install the requirements and package, run:

```
pip install -r requirements.txt
python setup.py install
```

## Dataset

To download the dataset used in the demo, run:


```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1tTj_bJ9nnc2ZfGB3cGZ_I3A5wqWK3aaD' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1tTj_bJ9nnc2ZfGB3cGZ_I3A5wqWK3aaD" -O van-hateren.zip 
rm -rf /tmp/cookies.txt
unzip van-hateren.zip
rm -r van-hateren.zip
mkdir datasets
mv van-hateren datasets
```


If your machine doesn't have wget, follow these steps: 
1. Download the zip file [here](https://drive.google.com/file/d/1tTj_bJ9nnc2ZfGB3cGZ_I3A5wqWK3aaD/view?usp=sharing).
2. Place the file in the top node of this directory, i.e. in `bispectral-networks/`.
3. Run:
    ```
    unzip van-hateren.zip
    rm -r van-hateren.zip
    mkdir datasets
    mv van-hateren datasets
    ```

## Demo

To view a simple demo, open the notebook at:

```
demo.ipynb
```



## License

This repository is licensed under the MIT License.  
