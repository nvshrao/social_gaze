## Data

We have retained gender and age information in the posts, as these are important for understanding social acceptance in models. No names or other personal identifying information were found in the dataset. Please use the data with care. You can download the data from [here](https://drive.google.com/drive/folders/1Dx7girZJOByU7GtoCvrZKCknK5b8U0V2?usp=sharing).  

You can paste the file `evaluate_rel_with_score.csv` into the `./data/` directory. 

## Usage

For Vanilla prompting:  
```bash
python vllm_aita.py phi-3-med
```

For SocialGaze prompting:  
```bash
python vllm_aita_delib.py phi-3-med
```
The generations will be saved into the `./results/` directory.
